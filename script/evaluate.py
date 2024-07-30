import os
import sys
import csv
import pprint
import random
import pickle

import numpy as np

from tqdm import tqdm

import torch
from torch.nn import functional as F

import torch_geometric.data     # Pre-load to bypass the hack from torchdrug
from torchvision import datasets
from torchdrug import core, utils, data, metrics
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from s3f import dataset, task, model, gvp


METRICS = ["spearmanr", "pearsonr", "mae", "rmse"]


def evaluate(pred, target):
    metric = {}
    for _metric in METRICS:
        if _metric == "mae":
            score = F.l1_loss(pred, target, reduction="mean")
        elif _metric == "rmse":
            score = F.mse_loss(pred, target, reduction="mean").sqrt()
        elif _metric == "spearmanr":
            score = metrics.spearmanr(pred, target)
        elif _metric == "pearsonr":
            score = metrics.pearsonr(pred, target)
        else:
            raise ValueError("Unknown metric `%s`" % _metric)
        
        metric[_metric] = score

    return metric


def graph_concat(graphs):
    if len(graphs) == 1:
        return graphs[0]
    graph = graphs[0].pack(graphs)
    if isinstance(graph, data.Protein):
        # residue graph
        _graph = data.Protein(edge_list=graph.edge_list, atom_type=graph.atom_type, bond_type=graph.bond_type, 
                                residue_type=graph.residue_type, atom_name=graph.atom_name, atom2residue=graph.atom2residue, 
                                residue_feature=graph.residue_feature, b_factor=graph.b_factor, bond_feature=None,
                                node_position=graph.node_position, num_node=graph.num_atom, num_residue=graph.num_residue,
        )
    else:
        # surface graph
        _graph = data.Graph(edge_list=graph.edge_list, num_node=graph.num_node, num_relation=1)
        with _graph.node():
            _graph.node_position = graph.node_position
            _graph.node_feature = graph.node_feature
            _graph.normals = graph.normals
    return _graph


def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0,seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0,model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0,mutation_position_relative-half_model_window), min(seq_len_wo_special,mutation_position_relative+half_model_window)]
    

def predict(cfg, task, dataset):
    dataloader = data.DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device(cfg.gpus[0])
    task = task.cuda(device)
    task.eval()
    seq_prob = []
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=device)
        with torch.no_grad():
            prob, sizes = task.inference(batch)
        cum_sizes = sizes.cumsum(dim=0)
        for i in range(len(sizes)):
            seq_prob.append(prob[cum_sizes[i]-sizes[i]:cum_sizes[i]])
    return seq_prob


def get_prob(seq_prob, mutations, offsets):
    i = 0
    preds = []
    targets = []
    last_sites = None
    for j, item in tqdm(enumerate(mutations)):
        sites, muts, target = item
        if j > 0 and sites != last_sites:
            i += 1

        node_index = torch.tensor(sites, dtype=torch.long)
        offset = offsets[i]
        node_index = node_index - offset
        mt_target = [data.Protein.residue_symbol2id.get(mut[-1], -1) for mut in muts]
        wt_target = [data.Protein.residue_symbol2id.get(mut[0], -1) for mut in muts]
        log_prob = torch.log_softmax(seq_prob[i], dim=-1)
        mt_log_prob = log_prob[node_index, mt_target]
        wt_log_prob = log_prob[node_index, wt_target]
        log_prob = mt_log_prob - wt_log_prob
        score = log_prob.sum(dim=0)

        preds.append(score)
        targets.append(target)
        last_sites = sites

    pred = torch.stack(preds)
    target = torch.tensor(targets).cuda()
    return pred, target
        

def load_dataset(csv_file, protein):
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin)
        fields = next(reader)
        mutations = []
        targets = []
        for i, values in enumerate(reader):
            for field, value in zip(fields, values):
                if field == "mutant":
                    mutations.append(value.split(":"))
                elif field == "DMS_score":
                    value = utils.literal_eval(value)
                    targets.append(value)
    
    def mutation_site(x):
        return [int(y[1:-1])-1 for y in x]

    mutations = [(tuple(mutation_site(mut)), mut, tar) for mut, tar in zip(mutations, targets)]
    mutations = sorted(mutations)
    sequences = []
    offsets = []
    for i, mut in enumerate(mutations):
        if i > 0 and mut[0] == mutations[i-1][0]:
            continue
        masked_seq = protein.clone()
        _mutation_site = mut[0]
        node_index = torch.tensor(_mutation_site, dtype=torch.long)

        # truncate long sequences and those only with substructures
        if os.path.basename(csv_file) == "POLG_HCVJF_Qi_2014.csv":
            start, end = 1981, 2225
        elif os.path.basename(csv_file) == "A0A140D2T1_ZIKV_Sourisseau_2019.csv":
            start, end = 290, 794
        elif os.path.basename(csv_file) == "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv":
            start, end = 119, 197       # keep high plddt part
        elif masked_seq.num_residue > 1022:
            seq_len = masked_seq.num_residue
            start, end = get_optimal_window(mutation_position_relative=mut[0][0], seq_len_wo_special=seq_len, model_window=1022)
        else:
            start, end = 0, masked_seq.num_residue
        node_index = node_index - start
        residue_mask = torch.zeros((masked_seq.num_residue, ), dtype=torch.bool)
        residue_mask[start:end] = 1
        masked_seq = masked_seq.subresidue(residue_mask)
        with masked_seq.graph():
            masked_seq.start = torch.as_tensor(start)
            masked_seq.end = torch.as_tensor(end)
        offsets.append(start)
        
        mask_id = task.model.sequence_model.alphabet.get_idx("<mask>")
        with masked_seq.residue():
            masked_seq.residue_feature[node_index] = 0
            masked_seq.residue_type[node_index] = mask_id
        sequences.append(masked_seq)

    return sequences, mutations, offsets


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Output dir: %s" % working_dir)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    summary = core.Configurable.load_config_dict(cfg.summary)
    num_assay = len(summary.ids)
    num_mutant = [int(summary.assay_dict[id]["DMS_number_single_mutants"]) for id in summary.ids]
    total_mutant = sum(num_mutant)
    if comm.get_rank() == 0:
        logger.warning("# total assays: %d, # total mutations: %d" % (num_assay, total_mutant))

    # pick the subset of DMS assays to be evaluated
    id_list = cfg.get("id_list", summary.ids)
    if "exclude_id_list" in cfg:
        exclude_id_list = set(cfg.exclude_id_list)
        id_list = [id for id in id_list if id not in exclude_id_list]

    if comm.get_rank() == 0:
        num_mutant = [int(summary.assay_dict[id]["DMS_number_single_mutants"]) for id in id_list]
        logger.warning("# assays: %d, # mutations: %d" % (len(id_list), sum(num_mutant)))

    task = core.Configurable.load_config_dict(cfg.task)
    task.preprocess(None, None, None)

    with open("results.csv", "w") as f:
        f.write("DMS_id,UniProt_ID,seq_len,DMS_number_single_mutants,%s\n" % (",".join(METRICS)))

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))['model']
        task.load_state_dict(model_dict)
    
    assay_result = {}
    for i, id in enumerate(id_list):
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Start evaluation on DMS assay %s" % id)
        
        # wild-type sequence
        sequence = summary.assay_dict[id]["target_seq"]
        protein = data.Protein.from_sequence(sequence, atom_feature=None, bond_feature=None)
        protein.view = "residue"

        # wild-type structure
        wild_types = []
        for _pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
            pdb_file = os.path.join(os.path.expanduser(cfg.structure_path), _pdb_file)
            wild_type = dataset.bio_load_pdb(pdb_file)[0]
            ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
            wild_type = wild_type.subgraph(ca_index)
            wild_types.append(wild_type)
        wild_type = graph_concat(wild_types)
        wild_type.view = "residue"

        # pdb range
        pdb_range = summary.assay_dict[id]["pdb_range"].split("-")
        start, end = int(pdb_range[0])-1, int(pdb_range[-1])
        with wild_type.graph():
            wild_type.start = torch.as_tensor(start)
            wild_type.end = torch.as_tensor(end)

        # surface graph
        if cfg.get("surface_path"):
            surf_graphs = []
            res2surfs = []
            for pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
                surf_graph_file = pdb_file.split(".")[0] + ".pkl"
                surf_graph_file = os.path.join(os.path.expanduser(cfg.surface_path), surf_graph_file)
                with open(surf_graph_file, "rb") as fin:
                    surf_dict = pickle.load(fin)
                surf_graph = dataset.load_surface(surf_dict)
                surf_graphs.append(surf_graph)
                res2surfs.append(torch.as_tensor(surf_dict["res2surf"]))
            surf_graph = graph_concat(surf_graphs)
            res2surf = torch.cat(res2surfs, dim=0)
            with wild_type.residue():
                wild_type.res2surf = res2surf
        else:
            surf_graph = None

        # mutants
        DMS_filename = summary.assay_dict[id]["DMS_filename"]
        csv_file = os.path.join(summary.path, DMS_filename)
        masked_sequences, mutations, offsets = load_dataset(csv_file, protein)
        if comm.get_rank() == 0:
            logger.warning("Number of masked sequences: %d" % len(masked_sequences))
            logger.warning("Number of mutations: %d" % len(mutations))

        _dataset = dataset.MutantDataset(masked_sequences, wild_type, surf_graph=surf_graph)
        seq_prob = predict(cfg, task, _dataset)
        pred, target = get_prob(seq_prob, mutations, offsets)
        result = evaluate(pred, target)

        if comm.get_rank() == 0:
            with open("%s" % DMS_filename, "w") as f:
                f.write(",mutant,mutated_sequence,DMS_score,model_score\n")
                pred = pred.cpu().numpy()
                target = target.cpu().numpy()
                for i in range(len(mutations)):
                    f.write(",%s,,%.6f,%.6f\n" % (":".join(mutations[i][1]), target[i], pred[i]))
            logger.warning(pretty.separator)
            logger.warning("Test results")
            logger.warning(pretty.line)
            logger.warning(pprint.pformat(result))
            with open("results.csv", "a") as f:
                f.write("%s,%s,%s,%s,%s\n" % (
                    id, 
                    summary.assay_dict[id]["UniProt_ID"], summary.assay_dict[id]["seq_len"],
                    summary.assay_dict[id]["DMS_number_single_mutants"],
                    ",".join([
                        "%.3f" % result[_metric] for _metric in METRICS
                    ])
                ))
        assay_result[id] = result

    if comm.get_rank() == 0:
        logger.warning(pretty.separator)
        logger.warning("Average results on all assays")
        logger.warning(pretty.line)
        logger.warning(pprint.pformat(utils.mean(utils.stack(list(assay_result.values())))))