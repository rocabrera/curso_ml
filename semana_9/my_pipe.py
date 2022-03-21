import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def create_pipe(model, numerical_columns, categorical_nominal_columns):

    # apenas as colunas com features categóricas
    one_hot_transformer = Pipeline([('onehot', OneHotEncoder(drop="first"))])
    preprocessor_categorical_nominal = ColumnTransformer(transformers=[('cat_nominal', 
                                                                         one_hot_transformer, 
                                                                         categorical_nominal_columns)])

    standirizer_transformer = Pipeline([
                                        ('imp_miss', SimpleImputer(strategy='constant')),
                                        ('standarizer', StandardScaler())
                                      ])
    preprocessor_numerical = ColumnTransformer(transformers=[('numerical', 
                                                              standirizer_transformer, 
                                                              numerical_columns)])

    processor_pipe = FeatureUnion([
                               ("preprocessor1", preprocessor_categorical_nominal),
                               ("preprocessor2", preprocessor_numerical)
                              ])

    pipe = Pipeline([("processor_pipe", processor_pipe),
                     ("rf", model)])

    return pipe
    # pipe.fit(X_train, y_train)
    # y_hat = pipe.predict(X_test)
    # print(classification_report(y_test, y_hat))