import TrainProcedure
import model

sampling_register = {
    'all_data': TrainProcedure.all_data,
    'all_data_xij': TrainProcedure.all_data_xij,
    'all_data_nobatch': None,
    'all_data_nobatch_xij': TrainProcedure.all_data_xij_no_batch
}

Rec_register = {
    'mf': model.RecMF
}

NOTE = """
    Any Var model should implement `allGamma` methods.
"""

Var_register = {
    'mf': model.VarMF_reg,
    'mf_xij': model.VarMF_xij,
    'mf_xij2': model.VarMF_xij2,
    'mf_itemper': model.VarMF_xij_item_personal,
    'mf_symper': model.VarMF_xij_Symmetric_personal,
    'mf_itemper_matrix' : None,
    'lgn': model.LightGCN,
    'lgn_xij': model.LightGCN_xij,
    'lgn_xij2': model.LightGCN_xij2,
    'lgn_itemper_single': model.LightGCN_xij_item_personal_single,
    'lgn_itemper_matrix': model.LightGCN_xij_item_personal_matrix
}



