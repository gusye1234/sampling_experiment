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
    Any Var model should implement `allGamma` methods for the test.
"""

Var_register = {
    'mf': model.VarMF_reg,
    'mf_itemper': model.VarMF_xij_item_personal,
    'lgn': model.LightGCN,
    'lgn_itemper_single': model.LightGCN_xij_item_personal_single,
    'lgn_itemper_matrix': model.LightGCN_xij_item_personal_matrix,
    'lgn_itemper_matrix_nohyper':model.LightGCN_xij_item_personal_matrix_nohyper
}



