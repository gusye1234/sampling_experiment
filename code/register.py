import TrainProcedure
import model

sampling_register = {
    "all_data_xij": TrainProcedure.all_data_xij,
    'all_data_nobatch_xij': TrainProcedure.all_data_xij_no_batch,
    'fast_sampling': TrainProcedure.sample_xij_no_batch
}

Rec_register = {
    'mf': model.RecMF
}

NOTE = """
    Any Var model should implement `allGamma` methods for the test.
"""

Var_register = {
    'lgn_itemper_matrix': model.LightGCN_xij_item_personal_matrix
}



