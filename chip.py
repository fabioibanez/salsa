import sys
import clip
import torch
from PIL import Image
from svm_utils import SalsaSVM
import pandas as pd
import numpy as np
import os
import pickle
from timeit import default_timer as now


# CLIP import stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

BASE_CHIP_DIR = "analysis/salsa/chips"


class ChipManager:
    def __init__(self, salsa, chips) -> None:
        """
        salsa is a Salsa object, chips is a dict of entity: Chip objects

        Args:
            salsa (_type_): _description_
            chips (_type_): _description_
        """
        # can access the chips through here!
        self._salsa = salsa
        # dict of entity: Chip objects
        self._chips = chips
        self.total_time_to_train_entitySVMs = 0

    def train_chips(self, labeled_df):
        """
        This function trains the chips on the labeled_df. This function
        is called in salsa.py after the SalsaSVM is trained. This is
        the core enabler of the chip system as it trains all of the svms.
        Note that this function relies on the Chip object's _train_chip,
        as it's a wrapper to train all the chips at once.

        Args:
            labeled_df (pandas dataframe): A dataframe with labels, and it is
            expected to have the labels for the entity in column:
            <ENTITY>_svm_label" as the function get_standard_embedding_list_entity
            expects this column.
        """
        for entity in self._chips:
            standard_embedding_list_entity = (
                self._salsa._embedding_svm.get_standard_embedding_list_entity(
                    labeled_df, entity
                )
            )
            (
                X_train,
                X_test,
                Y_train,
                Y_test,
            ) = self._salsa._embedding_svm.preprocess_embedding_data(
                standard_embedding_list_entity
            )
            entity_train_start = now()
            self._chips[entity]._train_chip(X_train, Y_train, X_test, Y_test)
            entity_train_end = now()
            self.total_time_to_train_entitySVMs += (
                entity_train_end - entity_train_start
            )
            self._save_chips()
            print(f"{entity} Chip recall is {self._chips[entity]._recall}")
            print(
                f"The total time to train the {entity} SVM is \
                    {entity_train_end - entity_train_start} seconds"
            )

    def _save_chips(self):
        """
        Save the chips to a pickle file
        """
        for entity in self._chips:
            # only want to save meaningful chips
            if self._chips[entity]._recall > 0.9:
                self._chips[entity]._save_chip()

    def apply_chips(self, df_to_write_out):
        """
        This function applies the chips to the df_to_write_out dataframe.
        It only applies positive filter chips.

        Args:
            df_to_write_out (pandas dataframe): df_to_write_out is either the
            results of the SalsaSVM or all of the images in the video in the
            case that the prompt is too specific and the SalsaSVM is not able
            to achieve a high enough recall.

        Returns:
            df_to_write_out_filtered (pandas dataframe): returns a dataframe with
            the predicted labels from the chips applied as a filter.
        """
        for entity in self._chips:
            # only apply the chip if it's good enough
            if self._chips[entity]._recall > 0.9:
                df_to_write_out = self._chips[entity]._apply_chip(
                    df_to_write_out
                )
        return df_to_write_out

    def define_chip(self):
        """
        This function allows the user to define a chip
        in the CLI. It allows a user to add a name for the
        chip and the ChipManager should handle the rest.
        """
        try:
            # Capture user input for the entity, dataset, and prompt
            entity = input(
                "Enter the name of the entity you want to track (e.g., 'car', 'human'): "
            )
            dataset = input(
                "Enter the name of the dataset (e.g., 'fast_and_furious_dataset'): "
            )
            prompt = input(
                "Enter the prompt to specify the chip (e.g., 'cars with red color'): "
            )

            # Validation could be added here if needed

            # Create a new Chip instance
            new_chip = Chip(entity, dataset, prompt)

            # Store the new chip in self._chips
            self._chips[entity] = new_chip

            print(f"Successfully defined a new chip for entity '{entity}'.")
            return entity

        except Exception as e:
            print(
                f"An error occurred while defining the chip: {e}",
                file=sys.stderr,
            )

    def apply_chips_dynamic_filter(self, df_to_write_out):
        """
        This function applies the chips filtering to the dataframe, and
        does so by checking whether or not an entity is present in the
        prompt. If it is present, then the chip will be applied as a positive
        filter. Otherwise, it will look at all of the chip objects saved from
        previous runs and apply them as a negative filter.

        Args:
            df_to_write_out (pandas dataframe): df_to_write_out is either the
            results of the SalsaSVM or all of the images in the video in the
            case that the prompt is too specific and the SalsaSVM is not able
            to achieve a high enough recall.

        Returns:
            df_to_write_out_filtered (pandas dataframe): returns a dataframe with
            the predicted labels from the chips applied as a filter.

        TODO: Implement
        """
        pass


class Chip:
    """
    The chip class is an abstraction of the larger Salsa system, which enables
    Salsa to be have cascading models for more fine-tuned results. The need for
    this was apparent when we realized that for 'crashes' in the fast and
    furious dataset our system performed well when being specific for
    crashes on fire, but was training a classifier on fire as opposed to
    classifier for 'crashes with fire'. We want to be able to have this
    level of speceficity while reducing the number of false positives a
    video analyst would have to go through.

    Novelties:
     - User is able to specify what they're trying to identify
     - Models can all be trained in parallel and then cascaded to produce
       more fine-tuned results
    """

    def __init__(self, entity, dataset, prompt) -> None:
        self._token = entity
        # each chip contains an SVM object as we're trying to train an SVM on a class by class basis
        self._svm_obj = SalsaSVM(dataset, prompt, entity)
        self._recall = 0

    def _save_chip(self):
        """
        save the chip object to a pickle file
        """
        filename = f"{self._token}_chip_obj.sav"
        path = os.path.join(BASE_CHIP_DIR, filename)
        pickle.dump(self, open(path, "wb"))

    def _load_chip(self):
        """
        load the chip object from a pickle file, expects the chip to
        exist otherwise it will not return anything.
        """
        filename = f"{self._token}_chip_obj.sav"
        path = os.path.join(BASE_CHIP_DIR, filename)
        if os.path.exists(path) == False:
            print("No chip object found, please train the chip first")
        else:
            return pickle.load(open(path, "rb"))

    def _train_chip(self, X_train, Y_train, X_test, Y_test):
        self._svm_obj._model = self._svm_obj.train_svm(X_train, Y_train)
        self._recall = self._svm_obj.get_recall(X_test, Y_test)

    def _predict(self, batch):
        batch_labels = self._svm_obj._model.predict(batch)
        return batch_labels

    def _apply_chip(self, df, positive=True):
        """
        the apply_chip function applies the chip on the remaining dataset,
        or in the case of scenario 3, where the targets in the images
        are too specific just run entirely on the entire dataset.

        Args:
            df (dataframe): either a df of the entire video or the df that is
            yielded by the main SalsaSVM. This dataframe is expected to already
            have the embeddings loaded into the 'embedding_array' column.

        Returns:
            modified df (dataframe): returns a dataframe with the predicated labels
        """
        batch_size = len(df) * 0.2
        predicted_labels = []
        for i in range(0, len(df), batch_size):
            batch = np.vstack(df.iloc[i : i + batch_size]["embedding_array"])
            batch_labels = self._predict(batch)
            predicted_labels.extend(batch_labels)
        df[f"{self._token}_label"] = predicted_labels

        if positive:
            filtered_df = df[df[f"{self.entity}_label"] == 1]
        elif not positive:
            filtered_df = df[df[f"{self.entity}_label"] == 0]
        return filtered_df
