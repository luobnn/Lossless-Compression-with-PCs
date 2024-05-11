import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.visualize as juice_vis
from pyjuice.model import TensorCircuit
from pyjuice.nodes import multiply, summate, inputs

import matplotlib.pyplot as plt
import BitVector as bv
import pickle
import numpy as np
import math
import time
from queue import Queue
from tqdm import trange

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class CompressProbCircuit:
    def clear_mar(self):
        """
        clearing the marginal of the whole PC (sum or input node to 0.0, prod node to -Inf)
        """
        for node, mar in self.mars.items():
            if node.is_sum() or node.is_input():
                self.mars[node] = torch.full_like(self.mars[node], 0.0)
            else:
                self.mars[node] = torch.full_like(self.mars[node], float("-inf"))

    def __init__(self, pc):
        def get_top_down_probs(pc):
            """
            get top down probability of all nodes (in log-likelihood form)
            and also get the empty marginal set of all nodes
            """
            device = torch.device("cuda:0")

            print("getting top down probs")
            ns = pc.root_ns
            top_down_prob = dict() # key type is juice.nodes.sum_nodes, value type is tensor, shape = [num_node_groups, group_size]
            mars = dict()
            num_nodes = []
            num_params = []
            def down(n):
                n_td_prob = top_down_prob[n]

                curr_num_nodes = n.num_node_groups * n.group_size
                num_nodes.append(curr_num_nodes)

                if n.is_sum():
                    
                    curr_num_params = n._params.size(0) * n._params.size(1) * n._params.size(2)
                    num_params.append(curr_num_params)

                    params = torch.log(n._params).to(device)
                    num_group_edges = n.num_node_groups * n.num_ch_node_groups
                    group_size = n.group_size
                    ch_group_size = n.ch_group_size
                    assert num_group_edges == params.shape[0] and group_size == params.shape[1] and ch_group_size == params.shape[2]
                    
                    child_td_prob = torch.full([n.num_ch_node_groups, ch_group_size], float("-inf")).to(device)
                    for group_edge_id in range(num_group_edges):
                        group_id = n.edge_ids[0, group_edge_id]
                        ch_group_id = n.edge_ids[1, group_edge_id]
                        for ch_node_id in range(ch_group_size):
                            param_tmp = params[group_edge_id, :, ch_node_id]
                            prob_tmp = n_td_prob[group_id, :]
                            down_prob_tmp = torch.logsumexp(param_tmp + prob_tmp, dim=0)
                            child_td_prob[ch_group_id, ch_node_id] = torch.logaddexp(child_td_prob[ch_group_id, ch_node_id], down_prob_tmp)
                    child = n.chs[0]
                    top_down_prob[child] = child_td_prob
                if n.is_prod():
                    for child in n.chs:
                        top_down_prob[child] = n_td_prob
                if n.is_input():
                    curr_num_params = n._params.size(0)
                    num_params.append(curr_num_params)

                for child in n.chs:
                    child_mar = torch.full([child.num_node_groups , child.group_size], 0.0).to(device)
                    mars[child] = child_mar
            
            print("Compiling CompressPC -- getting top down probs ...")
            top_down_prob[ns] = torch.zeros([1, 1]).to(device)
            mars[ns] = torch.zeros([1, 1]).to(device)
            eval_queue = Queue()
            eval_queue.put(ns)
            node_cnt = 0
            while not eval_queue.empty():
                curr_node = eval_queue.get()
                down(curr_node)
                #print("Compiling CompressPC -- getting top down probs :", node_cnt)
                node_cnt += 1
                for child in curr_node.chs:
                    if not child.is_input():
                        eval_queue.put(child)

            total_num_nodes = sum(num_nodes)
            total_num_params = sum(num_params)

            return [top_down_prob, mars, total_num_nodes, total_num_params, node_cnt] 
        def get_vars(pc):
            """
            get the variable order of the standard PC using DFS, 
            this is just the order following an pre-order traverse of the leaf node in PC
            (the right order is a in-order traverse of the vtree)
            """
            var_set = []
            def dfs(node):
                for child in node.chs:
                    if child.is_input():
                        var_set.append(child)
                    else:
                        dfs(child)
            root_node = pc.root_ns
            dfs(root_node)
            return var_set
        def get_evals(pc, var_set):
            """
            get the evaluation list of each variable according to the conditions in Algorithm 3
            """
            def dfs_need_eval(node, var_one, var_all, temp_eval_set):
                for child in node.chs:
                    if dfs_need_eval(child, var_one, var_all, temp_eval_set):
                        temp_eval_set.append(child)
                if node.scope.to_list() == list(var_one):
                    if node.is_input():
                        temp_eval_set.append(node)
                        return False
                    return True
                elif node.is_sum():
                    for child in node.chs:
                        if child in temp_eval_set:
                            return True
                elif node.is_prod():
                    if not var_one.issubset(node.scope.to_list()):
                        return False
                    for child in node.chs:
                        if var_all.issubset(child.scope.to_list()):
                            return False
                    return True
                else:
                    return False
            root_node = pc.root_ns
            var_all = set()
            eval_set = []
            print("Compiling CompressPC -- getting eval_list :")
            for var_id in trange(len(var_set)):
            #for (var_id, var_node) in enumerate(var_set):
                var_node = var_set[var_id]
                #print('\r'+"Compiling CompressPC -- getting eval_list :",var_id,"/",len(var_set),flush= True)
                var_one = set()
                temp_eval_set = []
                var_id = var_node.scope.to_list()[0]
                var_one.add(var_id)
                var_all.add(var_id)
                if dfs_need_eval(root_node, var_one, var_all, temp_eval_set):
                    temp_eval_set.append(root_node)
                eval_set.append(temp_eval_set)
            return eval_set
        def get_heads(eval_set):
            """
            get the head set of each variables which need to be summarized to calculate marginals
            """
            head_set = []
            for evalset_id in range(len(eval_set)):
                evals = eval_set[evalset_id]
                temp_head_set = set(evals)
                for node in evals:
                    children = set(node.chs)
                    temp_head_set = temp_head_set.difference(children)
                head_set.append(temp_head_set)
            return head_set
        def num_categories(pc):
            """
            get the maximun categories number of every input node
            """
            num_cats = 0
            for layer in pc.input_layer_group.layers:
                for cats in layer.metadata:
                    num_cats = max(cats, num_cats)
            return int(num_cats.item())

        [td_prob, marginals, total_num_nodes, total_num_params, num_vtree_nodes] = get_top_down_probs(pc)
        var_list = get_vars(pc)
        eval_list = get_evals(pc, var_list)
        head_list = get_heads(eval_list)
        self.total_num_nodes = total_num_nodes
        self.total_num_params = total_num_params
        self.num_vtree_nodes = num_vtree_nodes
        self.pc = pc
        self.var_order = var_list
        self.top_down_probs = td_prob
        self.head_list = head_list
        self.eval_list = eval_list
        self.num_vars = pc.num_vars
        self.num_cats = num_categories(pc)
        self.mars = marginals
        self.clear_mar()

    def prod_mar(self, prod_n):
        """
        calculate a prod node's marginals according to its children
        """
        device = torch.device("cuda:0")
        assert prod_n.is_prod()
        mar = torch.zeros([prod_n.num_node_groups, prod_n.group_size]).to(device)
        for group_id, mul_group in enumerate(prod_n.edge_ids):
            for child_id, child in enumerate(prod_n.chs):
                ch_mar = self.mars[child]
                mar[group_id, :] = mar[group_id, :] + ch_mar[mul_group[child_id], :]
        self.mars[prod_n] = mar

    def input_mar(self, input_n, token, cum_input):
        device = torch.device("cuda:0")
        assert input_n.is_input()
        scope_id = input_n.scope.to_list()[0]
        num_cats = input_n.dist.num_cats
        param = input_n._params
        mar = torch.zeros([input_n.num_node_groups, input_n.group_size]).to(device)
        for group_id in range(input_n.num_node_groups):
            for x in range(input_n.group_size):
                node_id = group_id * input_n.group_size + x
                if cum_input:
                    prob = sum(param[num_cats * node_id : num_cats * node_id + token])
                else:
                    prob = param[num_cats * node_id + token]
                mar[group_id, x] = torch.log(torch.Tensor([prob]))
        self.mars[input_n] = mar

    def sum_mar(self, sum_n):
        device = torch.device("cuda:0")
        assert sum_n.is_sum()
        params = sum_n._params.to(device)
        log_params = torch.log(params)
        num_node_groups = sum_n.num_node_groups
        num_ch_node_groups = sum_n.num_ch_node_groups
        num_group_edges = num_node_groups * num_ch_node_groups
        group_size = sum_n.group_size
        ch_group_size = sum_n.ch_group_size
        assert num_group_edges == params.shape[0] and group_size == params.shape[1] and ch_group_size == params.shape[2]

        ch_mar = self.mars[sum_n.chs[0]]
        tmp_mar = torch.log(torch.zeros([sum_n.num_node_groups, group_size])).to(device)
        for group_edge_id in range(num_group_edges):
            group_id = sum_n.edge_ids[0, group_edge_id]
            ch_group_id = sum_n.edge_ids[1, group_edge_id]
            A = log_params[group_edge_id, :, :]
            B = ch_mar[ch_group_id].repeat([group_size, 1])
            C = torch.logsumexp(A + B,dim = 1)
            tmp_mar[group_id] = torch.logaddexp(tmp_mar[group_id], C)
        self.mars[sum_n] = tmp_mar

    def forward(self, var_id, token, cum_input = False):
        """
        update the PC's state according to the input data 
        """
        input_n = self.var_order[var_id]
        self.input_mar(input_n, token, cum_input)
        evals = self.eval_list[var_id]
        for node in evals:
            if node.is_prod():        
                self.prod_mar(node)
            elif node.is_sum():
                self.sum_mar(node)

    def sum_up(self, var_id):
        """
        get the total target marginals
        """
        log_list = []
        for node in self.head_list[var_id]:
            td_prob = self.top_down_probs[node]
            mar = self.mars[node]
            assert mar.shape == td_prob.shape
            partial_ll = torch.logsumexp(mar + td_prob, dim = [0, 1])
            log_list.append(partial_ll)
        target_lls = torch.logsumexp(torch.Tensor(log_list), dim = 0)
        return target_lls.item()

    def marginal(self, tokens): 
        """
        input:
            tokens: size = [batch_size, num_vars]
        output:
            target_lls: size = [batch_size, num_vars] the log-likelihood of the non-cumulative marginal probability
            target_lls_cum: size = [batch_size, num_vars] the log-likelihood of the cumulative marginal probability
        """
        device = torch.device("cuda:0")

        batch_size = tokens.size(0)
        num_vars = self.num_vars
        assert tokens.size(1) == num_vars
        target_lls_cum = torch.zeros([batch_size, num_vars]).to(device)
        target_lls = torch.zeros([batch_size, num_vars]).to(device)

        for B in trange(batch_size):
            #print("clearing marginals ... computing the", B,"th example")
            self.clear_mar()
            for var_id in range(num_vars):
                scope_id = self.var_order[var_id].scope.to_list()[0]
                token = tokens[B, scope_id].item()
                self.forward(var_id, token, True)
                ll_cum = self.sum_up(var_id)
                target_lls_cum[B, var_id] = ll_cum
                #self.clear_partial_mar(var_id)
                self.forward(var_id, token)
                ll = self.sum_up(var_id)
                #print("example:",B+1,"/",batch_size,", variable:",var_id,"/",num_vars,", ll =",ll,", ll_cum =",ll_cum,", token =",token)
                target_lls[B, var_id] = ll
                
        return [target_lls, target_lls_cum]


"""
useful supplimentary functions
"""
def evaluate(pc, loader):
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        lls_total += lls.mean().detach().cpu().numpy().item()
    
    lls_total /= len(loader)
    return lls_total

def get_target_BV(intVal, size = 32):
    v = bv.BitVector(intVal = intVal)
    assert len(v) <= size
    while len(v) < size:
        zero = bv.BitVector(intVal = 0)
        v = zero + v
    return v

def bv2uint64(bv):
    if len(bv) == 0:
        return 0
    else:
        v = 0
        for idx in range(len(bv)):
            v = v << 1
            if bv[idx]:
                v = v | 1
        return v

"""
main compress/decompress functions
"""
def compress(CompPC, data, precision = 20):
    def get_codes_from_marginals(target_lls, target_lls_cum, precision = 20):
        assert target_lls_cum.shape == target_lls.shape

        rans_l = 1 << 31
        tail_bits = (1 << 32) - 1
        rans_prec_l = (rans_l >> precision) << 32
        log_prec = precision * math.log(2.0)

        # encode
        batch_size = target_lls_cum.size(0)
        num_examples = batch_size
        num_vars = target_lls_cum.size(1)
        states_head = [rans_l for _ in range(num_examples)]
        states_remain = [[] for _ in range(num_examples)]
        for ex_id in range(num_examples):
            for idx in range(num_vars-1, -1, -1):
            #for idx in range(num_vars):
                #print("Compressing... example :",ex_id + 1,"/",num_examples,", var :",idx+1,"/",num_vars)
                if idx >= 1:
                    ll_div = target_lls[ex_id, idx - 1].item()
                    ref_low = target_lls_cum[ex_id, idx].item() - ll_div
                    ref_ll = target_lls[ex_id, idx].item() - ll_div
                else:
                    ref_low = target_lls_cum[ex_id, idx].item()
                    ref_ll = target_lls[ex_id, idx].item()

                ref_low_int = math.ceil(math.exp(ref_low + log_prec))
                ref_ll_int = math.ceil(math.exp(ref_ll + log_prec))
                if ref_low_int % ref_ll_int == 0:
                    ref_low_int = ref_low_int - 1

                assert ref_ll_int > 0, "Please increase encoding precision (i.e., `precision`)."
                # ,"ll =",target_lls[ex_id, idx].item(), "ll_cum =",target_lls_cum[ex_id, idx].item()
                #,"ref_low_int =",ref_low_int, "ref_ll_int =",ref_ll_int
                #print("head before the",idx,"th var:",get_target_BV(intVal= states_head[ex_id], size = 64),"ll =",target_lls[ex_id, idx].item(), "ll_cum =",target_lls_cum[ex_id, idx].item())
                if states_head[ex_id] >= rans_prec_l * ref_ll_int:
                    #ans_temp = np.uint32(states_head[ex_id]) & tail_bits
                    #print("ori head:", get_target_BV(intVal = states_head[ex_id], size=64))
                    ans_temp = states_head[ex_id] & tail_bits
                    states_remain[ex_id].append(ans_temp)
                    states_head[ex_id] = (states_head[ex_id] >> 32)
                    
                    #print("head after the pop        :", get_target_BV(intVal = states_head[ex_id], size=64)) 
                    #print("into the remain list:", get_target_BV(intVal = ans_temp))
                a1 = (states_head[ex_id] // ref_ll_int) << precision 
                a2 =  (states_head[ex_id] % ref_ll_int) + ref_low_int
                states_head[ex_id] = a1 + a2
        codes = []
        for ex_id in range(num_examples):
            compressed_bv = bv.BitVector(intVal = states_head[ex_id])
            for (idx, int32) in enumerate(states_remain[ex_id]):
                if idx == 0:
                    compressed_bv = get_target_BV(intVal = int32)
                else:
                    tmp_bv = get_target_BV(intVal = int32)
                    compressed_bv = compressed_bv + tmp_bv

            head_bv = bv.BitVector(intVal = states_head[ex_id])
            compressed_bv = compressed_bv + head_bv
                
            codes.append(compressed_bv)
            #print(len(compressed_bv))
        return codes

    [target_lls, target_lls_cum] = CompPC.marginal(tokens = data)
    codes = get_codes_from_marginals(target_lls, target_lls_cum, precision)
    return codes

def decompress(CompPC, codes, precision = 20):
    batch_size = len(codes)
    num_vars = CompPC.num_vars
    target_data = torch.zeros([batch_size, num_vars])
    max_num_binary_search = math.ceil(math.log2(num_cats))
    rans_l = 1 << 31
    tail_bits = (1 << 32) - 1
    rans_prec_l = ((rans_l >> precision) << 32)
    prec_tail_bits = (1 << precision) - 1
    prec_val = 1 << precision
    log_prec = precision * math.log(2.0)

    states_head = [0 for _ in range(batch_size)]
    states_remain = [[] for _ in range(batch_size)]

    for ex_id in range(batch_size):
        print(codes[ex_id])
        siz = len(codes[ex_id])
        head_len = 2
        #if codes[ex_id][siz-1]:
        #    head_len = 2
        s_len = math.ceil(siz / 32) - head_len
        states_remain[ex_id] = [0 for _ in range(s_len)]
        for idx in range(s_len):
            print(codes[ex_id][ (idx << 5) : ((idx + 1) << 5)])
            states_remain[ex_id][idx] = bv2uint64(codes[ex_id][ (idx << 5) : ((idx + 1) << 5)])
       
        head_bv = codes[ex_id][(s_len << 5) : siz]
        print("head", head_bv)
        states_head[ex_id] = bv2uint64(head_bv)

    for ex_id in range(batch_size):
        CompPC.clear_mar()
        lls_div = 0.0
        for (var_id, input_n) in enumerate(CompPC.var_order):
            low = 0
            high = CompPC.num_cats
            if states_head[ex_id] & prec_tail_bits == 0:
                lls_ref = float("-Inf")
            else:
                lls_ref = math.log((states_head[ex_id] & prec_tail_bits)) - log_prec
            lls_cdf = float("-inf")

            for _ in range(10):
                mid = (low + high) // 2
                CompPC.forward_one(var_id, mid, cum_input = True)
                lls_buffer = CompPC.sum_up(var_id)
                check_lls = lls_div + lls_ref
                cond = lls_buffer - lls_div < lls_ref
                if cond:
                    low = mid
                    lls_cdf = lls_buffer
                else:
                    high = mid

            scope_id = CompPC.var_order[var_id].scope.to_list()[0]
            target_data[ex_id, scope_id] = mid
            print("ex_id :",ex_id,", var_id =",var_id,", scope =",scope_id,"decom data =", mid, "real data =", ans_data[ex_id, scope_id].item())

            CompPC.forward_one(var_id, mid, cum_input = False)
            lls_new = CompPC.sum_up(var_id)

            #update_rans_codes(lls_cdf, lls_new, lls_div, log_prec, ref_low_int, ref_ll_int)
            val1 = math.ceil(math.exp(lls_cdf - lls_div + log_prec))
            val2 = math.ceil(math.exp(lls_new - lls_div + log_prec))
            if val1 % val2 == 0:
                val1 = val1 - 1
            ref_low_int = val1
            ref_ll_int = val2

            cf = (states_head[ex_id] & prec_tail_bits)
            states_head[ex_id] = (states_head[ex_id] >> precision) * ref_ll_int + cf - ref_low_int
            if states_head[ex_id] < rans_l:
                if len(states_remain[ex_id]) != 0:
                    tmp_remain = states_remain[ex_id].pop()
                    states_head[ex_id] = (states_head[ex_id] << 32) | tmp_remain
                else:
                    states_head[ex_id] = states_head[ex_id] << 32
            #print("ex_id :",ex_id,", scope :",scope_id,", decompressed data :", target_data[ex_id, scope_id].item(), "real data", ans_data[ex_id, scope_id].item())
            lls_div = lls_new
    return target_data

"""
testing functions
"""

def compress_test():
    """
    compress MNIST just One batch
    """

    with open('CompPC_for_mnist_new.pkl', 'rb') as file_CPC:
        CompPC = pickle.load(file_CPC)
    device = torch.device("cuda:0")
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)
    test_data = test_dataset.data.reshape(10000, 28*28)
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = 30, #可以修改
        shuffle = False,
        drop_last = True
    )
    for batch in test_loader:
        data = batch[0].to(device)

    codes = compress(CompPC, data)
    bitlen = [len(x) for x in codes]
    return
    
def decompress_using_states(CompPC, states_head, states_remain, ans_data, precision = 20):
    """
    Using states_head and states_remain to decompress
    """

    batch_size = len(states_head)
    num_vars = CompPC.num_vars
    target_data = torch.zeros([batch_size, num_vars])
    max_num_binary_search = math.ceil(math.log2(num_cats))
    rans_l = 1 << 31
    tail_bits = (1 << 32) - 1
    rans_prec_l = ((rans_l >> precision) << 32)
    prec_tail_bits = (1 << precision) - 1
    prec_val = 1 << precision
    log_prec = precision * math.log(2.0)

    for ex_id in range(batch_size):
        CompPC.clear_mar()
        lls_div = 0.0

        for (var_id, input_n) in enumerate(CompPC.var_order):
            low = 0
            high = CompPC.num_cats
            if states_head[ex_id] & prec_tail_bits == 0:
                lls_ref = float("-Inf")
            else:
                lls_ref = math.log((states_head[ex_id] & prec_tail_bits)) - log_prec
            lls_cdf = float("-inf")

            for _ in range(10):
                mid = (low + high) // 2
                CompPC.forward_one(var_id, mid, cum_input = True)
                lls_buffer = CompPC.sum_up(var_id)
                check_lls = lls_div + lls_ref
                cond = lls_buffer - lls_div < lls_ref
                if cond:
                    low = mid
                    lls_cdf = lls_buffer
                else:
                    high = mid

            scope_id = CompPC.var_order[var_id].scope.to_list()[0]
            target_data[ex_id, scope_id] = mid
            print("ex_id :",ex_id,", var_id =",var_id,", scope =",scope_id,"decom data =", mid, "real data =", ans_data[ex_id, scope_id].item())

            CompPC.forward_one(var_id, mid, cum_input = False)
            lls_new = CompPC.sum_up(var_id)

            #update_rans_codes(lls_cdf, lls_new, lls_div, log_prec, ref_low_int, ref_ll_int)
            val1 = math.ceil(math.exp(lls_cdf - lls_div + log_prec))
            val2 = math.ceil(math.exp(lls_new - lls_div + log_prec))
            if val1 % val2 == 0:
                val1 = val1 - 1
            ref_low_int = val1
            ref_ll_int = val2

            cf = (states_head[ex_id] & prec_tail_bits)
            states_head[ex_id] = (states_head[ex_id] >> precision) * ref_ll_int + cf - ref_low_int
            if states_head[ex_id] < rans_l:
                if len(states_remain[ex_id]) != 0:
                    tmp_remain = states_remain[ex_id].pop()
                    states_head[ex_id] = (states_head[ex_id] << 32) | tmp_remain
                else:
                    states_head[ex_id] = states_head[ex_id] << 32
            #print("ex_id :",ex_id,", scope :",scope_id,", decompressed data :", target_data[ex_id, scope_id].item(), "real data", ans_data[ex_id, scope_id].item())
            lls_div = lls_new
    return target_data

def pre_test():
    """
    load the trained Compress PC
    """
    import sys
    sys.setrecursionlimit(3000)
    pc_file_name = f"CIFAR10_8_t100.jpc"
    device = torch.device("cuda:0")
    # 预处理CompPC
    ns = juice.io.load(pc_file_name)
    pc = TensorCircuit(ns)
    pc.to(device)
    CompPC = CompressProbCircuit(pc)
    with open('CIFAR10_8_t100.pkl', 'wb') as file_CPC:
        pickle.dump(CompPC, file_CPC)

def main_test(start_batch_id = 0):
    """
    load the compress pc, 
    """
    with open('CIFAR10_8_t100.pkl', 'rb') as file_CPC:
        CompPC = pickle.load(file_CPC)
    
    device = torch.device("cuda:0")
    test_dataset = torchvision.datasets.CIFAR10(root = "./examples/data", train = False, download = True)
    CHANNEL = 0
    test_data = torch.Tensor(test_dataset.data[:, :, :, CHANNEL].reshape(10000, 32*32)).long().to(device)
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = 50, #可以修改
        shuffle = False,
        drop_last = True
    )

    target_codes = []
    ans_data = []
    check_set = [] # 可以修改
    num_batches = len(test_loader)
    for (batch_id, batch) in enumerate(test_loader):
        if batch_id < start_batch_id:
            continue
        data = batch[0].to(device)
        print("Batch :",batch_id,"/",num_batches)
        batch_codes = compress(CompPC, data)
        target_codes = target_codes + batch_codes
        ans_data = ans_data + data.tolist()

        batch_bitlen = [len(x) for x in batch_codes]
        print(batch_bitlen)

        if batch_id in check_set:
            batch_decompress_data = decompress(CompPC, batch_codes)
            assert batch_decompress_data == data ,"ERROR: the compress/decompress is wrong"

    with open('Compressed MNIST', 'wb') as file_MNIST:
        pickle.dump([target_codes, ans_data])

def plot():
    pc_file_name = f"CIFAR10_128_0_fullbatch.jpc"
    device = torch.device("cuda:0")
    ns = juice.io.load(pc_file_name)

    plt.figure()
    juice_vis.plot_pc(ns, node_id=False, node_num_label=False)
    plt.savefig("CIFAR10_128_0_fullbatch.png")

if __name__ == "__main__":
    pre_test()
    main_test()
    print("end")