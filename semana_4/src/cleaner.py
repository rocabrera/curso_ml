import pandas as pd

def assign_correct_types(df:pd.DataFrame, is_train_dataset=True) -> pd.DataFrame:
    
    # Questões com tp
    tp_columns = df.filter(regex="TP_").columns
    tp_exclude = ["TP_SEXO"]
    tp_mapping = {column:("float32" if df.loc[:,column].hasnans else "uint8") for column in tp_columns if column not in tp_exclude}

    # Questões com sg
    sg_columns = df.filter(regex = "SG_").columns
    sg_mapping = {column:"category" for column in sg_columns}
    
    # Questões com NO
    no_columns = df.filter(regex="NO_M").columns
    no_mapping = {column:"category" for column in no_columns}
    
    
    # Questões com in
    in_columns = df.filter(regex="IN_").columns
    in_mapping = {column:("float32" if df.loc[:,column].hasnans else "bool") for column in in_columns}

    # Dados do questionário ecônomico
    numerical_ranked_columns = ["Q005"]
    numerical_ranked_mapping = {column:"uint8" for column in numerical_ranked_columns}

    object_ranked_columns = ["Q001", "Q002", "Q003", "Q004", "Q006", 
                             "Q007", "Q008", "Q009", "Q010", "Q011", 
                             "Q012", "Q013", "Q014", "Q015", "Q016", 
                             "Q017", "Q019", "Q022", "Q024"]
    object_ranked_mapping = {column:"category" for column in object_ranked_columns}

    object_binary_columns = ["Q018", "Q020", "Q021", "Q023", "Q025"]
    object_binary_mapping = {column:"category" for column in object_binary_columns}

    
    # Outras variáveis 
    other_mapping = {"NU_IDADE":"float32",
                     "TP_SEXO":"category",
                     "CO_ESCOLA":"float32"}
    
    target_mapping = {}
    
    if is_train_dataset:
        # Variáveis targets
        target_columns = df.filter(regex="NU_NOTA").columns
        target_mapping = {column:"float32" for column in target_columns}

    types_mapping = {**tp_mapping, 
                     **sg_mapping,
                     **no_mapping,
                     **in_mapping, 
                     **numerical_ranked_mapping,
                     **object_ranked_mapping,
                     **object_binary_mapping,
                     **other_mapping,
                     **target_mapping}

    return df.astype(types_mapping)