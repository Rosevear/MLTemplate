import config
import utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def create_classifier_pipeline(clf, data, sparse=False):
    """
    Embed the classifier clf into a machine learning pipeline

    Setting sparse to false will tell the 1-hot encoder to return matrices in a compressed format for computational efficiency: See https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
    
    data is used to extract the heading columns and their positions. It can also be used to compute statistics over the data that may be needed to pass into the pipeline
    """

    column_headers = data.columns.tolist()

    if config.CALIBRATE_PROBABILITY:
        cv_procedure = utils.get_cv_procedure()
        clf = CalibratedClassifierCV(clf, cv_procedure, config.CALIBRATION_METHOD)


    # Find all of the values in the training set for each type of categorical variable
    categories = [np.sort(data[category].unique()) for category in config.CATEGORICAL_COLUMNS]
    
    # This column transformer uses a 1-hot encoder for categorical data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    one_hot_encoding_step = ('One Hot Encoding Transform for Categorical Data', OneHotEncoder(categories=categories,
        sparse=sparse, dtype=np.float, handle_unknown='ignore'), utils.get_column_positions(column_headers, config.CATEGORICAL_COLUMNS))

    # And a Standard Scaler for numerical interval data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    standardization_step = ('Standardization For Interval Data',
                            StandardScaler(), utils.get_column_positions(column_headers, config.NUMERICAL_COLUMNS))

    # Transformer utility class to encode the inputs of different columns: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    transformer = ColumnTransformer(transformers=[
                                    one_hot_encoding_step, standardization_step],  remainder='passthrough')

    datatype_transform = FunctionTransformer(utils.convert_array_to_dtype)

    pipeline = Pipeline(steps=[(config.COLUMN_TRANSFORMER_STEP_NAME, transformer),
                                ('datatypeTransform', datatype_transform),
                               (config.CLASSIFIER_STEP_NAME, clf)])

    return pipeline


def create_keras_model(input_dim):
    """
    The tensorflow keras module provides a scikit-learn classifier wrapper class that implements the scikit-learn API. See here: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/wrappers/scikit_learn.py#L191-L310
    """
	
    # Define the model structure 
    model = Sequential()
    model.add(Dense(units=100, input_dim=input_dim, use_bias=True, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
    return model


def get_keras_classifier_pipeline(data):
    """
    Keras Classifier: https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier
    """

    # Infer the shape of the feature vector size after passing in some testing through the pipeline ot transform it
    if config.INFER_KERAS_INPUT_SHAPE:
        spy = utils.Pipeline_Spy()
        pipeline = create_classifier_pipeline(spy, data)
        
        # Need to represent the single data sample as a 1 by num_features array, not a 1-dimensional vector num_features long
        data_sample = np.array(data.iloc[1, :])[np.newaxis, ...] 
        print("Original data shape: {}".format(data_sample.shape))
        
        feature_vector_transformed = pipeline.fit_transform(data_sample)[0]
        print("Transformed data shape: {}".format(feature_vector_transformed.shape))
        
        feature_vector_input_length = len(feature_vector_transformed)

        print("Inferred feature vector length for Keras model: {}".format(
            feature_vector_input_length))
    else:
        feature_vector_input_length = config.KERAS_INPUT_SHAPE

    clf = KerasClassifier(build_fn=create_keras_model,
                          input_dim=feature_vector_input_length, epochs=150, batch_size=32)

    return create_classifier_pipeline(clf, data)


def get_passive_agressive_classifier_pipeline(data):
    """
    https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier
    """

    clf = PassiveAggressiveClassifier()

    return create_classifier_pipeline(clf, data)

def get_sgd_classifier_pipeline(data):
    """
    https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    """

    clf = SGDClassifier(loss='hinge', penalty='elasticnet')

    return create_classifier_pipeline(clf, data)


def get_naive_bayes_classifier_pipeline(data):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn-naive-bayes-gaussiannb
    """

    clf = GaussianNB()

    return create_classifier_pipeline(clf, data)

def get_dummy_classifier_pipeline(data):
    """
    Dummy Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier
    """

    clf = DummyClassifier()

    return create_classifier_pipeline(clf, data)


def get_MLP_classifier_pipeline(data):
    """
    MLP Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    clf = clf = MLPClassifier(hidden_layer_sizes=(100,),
                              activation='relu',
                              solver='adam',
                              alpha=0.0001,  # L2 regularization parameter
                              batch_size='auto',
                              learning_rate='adaptive',
                              learning_rate_init=0.001,
                              power_t=0.5,  # Only used for invscaling option of learning_rate
                              momentum=0.9,  # Only used for SGD optimizer
                              nesterovs_momentum=True,  # Only used for SGD and momentum > 0
                              beta_1=0.9,  # Both betas are paramters for the Adam solver
                              beta_2=0.999,
                              epsilon=1e-8,  # Adam optimizer numerical stability constant
                              max_iter=200,
                              early_stopping=True,
                              verbose=config.VERBOSE,
                              shuffle=True,
                              random_state=config.RANDOM_SEED)


    return create_classifier_pipeline(clf, data)


def get_KNN_classifier_pipeline(data):
    """
    #K-Nearest Neighbours: https://scikit-learn.org/stable/modules/neighbors.html#id6
    """
    clf = KNeighborsClassifier(n_neighbors=5)

    return create_classifier_pipeline(clf, data)


def get_DT_classifier_pipeline(data):
    """
     Decision Tree Classifier: https://scikit-learn.org/stable/modules/tree.html
    """

    clf = DecisionTreeClassifier(max_depth=20,
                                 random_state=config.RANDOM_SEED)

    return create_classifier_pipeline(clf, data)


def get_logit_classifier_pipeline(data):
    """
    Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    clf = LogisticRegression()

    return create_classifier_pipeline(clf, data)


def get_perceptron_classifier_pipeline(data):
    """
    Perceptron: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
    """
    
    clf = Perceptron(penalty='l1',
                     alpha=0.0001,
                     fit_intercept=True,
                     max_iter=1000,
                     tol=1e-3,
                     eta0=1,
                     early_stopping=True,
                     validation_fraction=0.1,
                     n_iter_no_change=5,
                     random_state=config.RANDOM_SEED)

    return create_classifier_pipeline(clf, data)
