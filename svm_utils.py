import os
import mping
import cv2
import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

base_analysis_dir = "analysis"
base_labeled_df_dir = os.path.join(base_analysis_dir, "salsa/labeled_df")
base_model_dir = os.path.join(base_analysis_dir, "salsa/models")


class SalsaSVM:
    """
    By defualt the SalsaSVM class is nonspecific and on a query to query basis. This means that
    there is no particular 'class' that the SVM is trying to identify - it is only trying to discern
    the differences between the user's preferences and the rest of the dataset. This is useful for
    fine-tuning frozen models, but it is not useful for identifying a particular class and this could
    have many implications. For example if you can train SVM's on a class by class basis, then you can
    also cascade these models to produce more fine-tuned results or even use these models as negative
    filters for other classes that are unrelated.
    """

    def __init__(self, dataset, prompt, token=None):
        self._dataset = dataset
        self._prompt = prompt
        self._token = token

        self._svm_obj = None
        self._model = None
        self._recall = 0

        model_identifier = self._token if self._token else self._prompt
        """ TODO(feature): add an interface to handle the case where the model already exists! \
            this should not be a part of this interface.
        """
        # if self._model_exists(model_identifier):
        #     self._svm_obj = self._load_obj(model_identifier)
        #     self._recall = self._svm_obj.recall
        #     self._model = self._svm_obj.model

    def _model_exists(self, class_name):
        filename = f"{class_name}_svm_obj.sav"
        path = os.path.join(base_model_dir, filename)
        return os.path.exists(path)

    def _load_obj(self, class_name):
        filename = f"{class_name}_svm_obj.sav"
        path = os.path.join(base_model_dir, filename)
        with open(path, "rb") as file:
            _svm_obj = pickle.load(file)
        return _svm_obj

    def load_data(self):
        pd_df = pd.read_parquet(
            os.path.join(
                base_labeled_df_dir,
                f"labeled_df_{self._dataset}_{self._prompt}.parquet",
            )
        )
        pd_df["encoded_label"] = [
            1 if label else 0 for label in pd_df["svm_label"]
        ]

        return pd_df

    def train_svm(self, X_train, y_train):
        """
        Train the SVM on the training data
        """
        clf = SVC()
        if len(np.unique(y_train)) == 1:
            print("Only one class detected, skipping training")
            return clf
        clf.fit(X_train, y_train)
        return clf

    def save_model(self, svm_obj, class_name):
        """
        save the model
        """
        filename = f"{class_name}_svm_obj.sav"
        path = os.path.join(base_model_dir, filename)
        pickle.dump(svm_obj, open(path, "wb"))

    def _is_fitted(self):
        """
        In our pipeline of chips we need to know if the model is fitteed or not
        otherwise we might call functions that rely on our models being fitted.
        A user may not really care about one particular chip and it ends up
        only being uni-class (only False, and our pipeline will ignore (i.e not fit)
        those types of chips

        Returns:
            bool: representing whether or not a model is fitted.
        """
        try:
            check_is_fitted(self._model)
            return True
        except NotFittedError:
            return False

    def get_recall(self, X_test, y_test):
        if self._is_fitted() is False:
            return 0
        y_pred = self._model.predict(X_test)
        return recall_score(y_test, y_pred)

    class PixelsSalsaSVMDataProcessor:
        """
        ImageSalsaSVM class has the infrastructure to create an SVM, and train and test on images (using frames).
        See EmbeddingSalsaSVM for the same thing but with embeddings.
        """

        def standardize_image(self, image):
            """
            Save this standard image somewhere
            """
            standardized = cv2.resize(image, (256, 256))
            return standardized

        def get_standard_image_list(self, dataframe):
            """
            Returns a list of images that have been standardized
            """
            standard_list = []

            for _, row in dataframe.iterrows():
                frameuri = row["frameuri"]
                frame = mping.imread(frameuri)
                # NOTE: this is a tuple of (image, label)
                standard_list.append(
                    (
                        self.standardize_image(frame),
                        row["encoded_label"],
                    )
                )

            return standard_list

        def preprocess_pixel_data(self, standard_list, test_size=0.2, seed=7):
            """
            Preprocess the data: flatten the images and split into training and testing sets.
            """
            # NOTE: image needs to be flattend
            X = [x[0].flatten() for x in standard_list]
            Y = [y[1] for y in standard_list]

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = sklearn.model_selection.train_test_split(
                X, Y, test_size=test_size, random_state=seed
            )

            return X_train, X_test, y_train, y_test

    class EmbeddingSalsaSVMDataProcessor:
        # get the dataframe with load_dataframe
        # TODO: consolidate these two functions
        def get_standard_embedding_list(self, dataframe, entity=""):
            """
            Input: labeled dataframes with embeddings in ["embedding_array"] and label in ["svm_label"]
            Returns a list of embeddings that have been standardized, and labels in the form of a tuple
            """
            standard_list = []
            for _, row in dataframe.iterrows():
                embedding_array = row["embedding_array"]
                standard_list.append((embedding_array, row["svm_label"]))
            return standard_list

        def get_standard_embedding_list_entity(self, dataframe, entity):
            standard_list = []
            for _, row in dataframe.iterrows():
                embedding_array = row["embedding_array"]
                standard_list.append(
                    (embedding_array, row[f"{entity}_svm_label"])
                )
            return standard_list

        def preprocess_embedding_data(
            self, standard_list, test_size=0.2, seed=7
        ):
            """
            Preprocess the data: flatten the images and split into training and testing sets.
            """
            X = [x[0] for x in standard_list]
            Y = [y[1] for y in standard_list]

            (
                X_train,
                X_test,
                Y_train,
                Y_test,
            ) = sklearn.model_selection.train_test_split(
                X, Y, test_size=test_size, random_state=seed
            )

            return X_train, X_test, Y_train, Y_test
