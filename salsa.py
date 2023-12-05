import os
import sys
import matplotlib
from similarity_search_class import SimilaritySearch
from svm_utils import SalsaSVM
from timeit import default_timer as now
import numpy as np
import logging
from nltk.tag import pos_tag
from chip import Chip, ChipManager
from image_display import ImageDisplay

# sys.path.append needs to go before the import
# of compute_accuracy_retrieval_simple
sys.path.append("/opt/viva/analysis")
from compute_accuracy_retrieval_simple import *

matplotlib.use("TkAgg")  # Use the TkAgg backend to handle image display

base_analysis_dir = "analysis"
base_results_df_embedding_dir = os.path.join(
    base_analysis_dir, "salsa/results_df_embedding"
)

num_frames = 30
MMR_ALPHA = 0.0

logging.basicConfig(level=logging.DEBUG)  # Set logging to DEBUG level


class Salsa:
    """
    Salsa allows for fine-tuning of general video analytics
    systems that make use of Visual Language Model embeddings.
    Salsa creates a user-defined pipeline for fine-tuning
    the results yielded by the VIVA system. The user is able
    choose the images that they want to see more of and also
    define the entities that they want to see more of utilizing
    the chip abstraction. Chips are a wrapper on the SalsaSVM
    class which in parallel with the general SalsaSVM trains
    SVM's on a class by class basis.

    Args:
        prompt (str): The prompt that the user is interested in
                          fine-tuning the results for
        dataset (str): The dataset that the user is interested in
                          fine-tuning the results for
        method (str): The method that the user is interested in
                      currently only embeddings are supported but
                      pixel-based methods will be supported in the
                      near future.
        mode (str): The mode that the user is interested in
                    currently only exploitation is supported but
                    exploration will be supported in the near future.
        additional_entities (list, optional): A list of additional
                                                entities that the user
                                                can define. These are
                                                the classes that Salsa
                                                can train SVM's on. Defaults to [].
    TODO:
        * Add support for pixel-based methods
        * Add support for exploration mode
    """

    def __init__(
        self,
        prompt: str,
        dataset: str,
        method: str,
        mode: str,
        additional_entities=[],
    ):
        self._prompt = "".join(word for word in prompt.split("_"))
        self._dataset = dataset
        self._similarity_search_obj = SimilaritySearch(self._prompt, dataset)

        self._svm_obj = SalsaSVM(dataset, self._prompt)
        self._model_threshold = 0.9
        self._method = method

        # exploration or exploitation
        self._mode = mode
        self._embedding_svm = self._svm_obj.EmbeddingSalsaSVMDataProcessor()

        # NOTE: all of the stuff is embedding-based now
        self._pixel_svm = self._svm_obj.PixelsSalsaSVMDataProcessor()

        self.entities = self._get_proper_nouns(prompt.replace("_", " "))
        self.entities.extend(additional_entities)
        # remove duplicates if any
        self.entities = list(set(self.entities))

        self._chip_manager = ChipManager(
            self,
            {
                entity: Chip(entity, dataset, self._prompt)
                for entity in self.entities
            },
        )

    def _get_proper_nouns(self, text: str) -> list:
        """
        This function is used to get the proper nouns from the prompt.
        This allows Salsa to automatically detect the entities that
        it can train specialized SVM's on.

        Args:
            text (str): string of space separated words

        Returns:
            lsit: list of proper nouns
        """
        tagged_sent = pos_tag(text.split())
        propernouns = [
            word
            for word, pos in tagged_sent
            if pos == "NNP" or pos == "NNS" or pos == "NNPS" or pos == "NN"
        ]
        return propernouns

    def _naive_filter(self, df_to_write_out, jump=0):
        """
        This function is to filter out frames that are too similiar
        to other frames by enforcing an arbitrary jump between frames.
        This function is used in the _show_me_more_* functions.

        Args:
            df_to_write_out (dataframe): contains the positive results
                                         of the svm
            jump (int, optional): The minimum distance between frames.
                                  Defaults to 10.

        Returns:
            dataframe: filtered dataframe
        """
        return df_to_write_out.iloc[::jump, :]

    def _show_me_more_embedding(self):
        """
        This is the current backbone of the Salsa pipeline. It is
        the function that is called when the user wants to see more
        and use embeddings as the common representation. In this function
        a few things happen:

        1) The user is shown the top_k_frames (default is 30) and is
           prompted to label them as either desired (True) or not (False)

        2) The user is then prompted to more granularly choose which
           entities are in the frame. This logic is all handled in the
           ImageDisplay class.

        3) The user's choices are then used to train the general SalsaSVM
           and the chips - which rely on the labeled entity data.

        4) Based on the results of the user's choices the user is then
           shown more frames based on an embedding similarity search
           and the process repeats. At each stage getting more
           data to train the general SalsaSVM and the chips.
        """
        # initialize state
        best_recall = 0
        accepted_frames_list_input = []

        # load in without duplicates!
        all_images_df = (
            self._similarity_search_obj.SVMImageIO.get_all_images_df()
        )

        # on iter0 Salsa uses the results from the video analytics platform
        top_k_frames = self._similarity_search_obj.SVMImageIO.get_parquet(
            self._similarity_search_obj.SVMImageIO.results_pq
        )
        total_time_to_train_SalsaSVM = 0

        # add any chips manually
        continue_adding = True
        while continue_adding:
            continue_adding = input("Do you want to add any chips? (y/n)")
            if continue_adding == "n":
                break
            entity = self._chip_manager.define_chip()
            continue_adding = input("Do you want to add another chip? (y/n)")
            if continue_adding == "n":
                continue_adding = False
            elif continue_adding == "y":
                continue_adding = True
            else:
                print("Please enter 'y' or 'n'")
            self.entities.append(entity)

        self.entities = list(set(self.entities))

        while (
            self._svm_obj._recall < self._model_threshold
            and self._similarity_search_obj.batch_num < 5
        ) or True:
            logging.info(
                f"Currently doing similarity search on {len(top_k_frames)} frames"
            )
            print(f"self.entities", self.entities)
            image_display = ImageDisplay(top_k_frames, self.entities)
            image_display.start()

            # boolean list of accepted frames for this batch; updates on a batch-basis
            accepted_frames_list_input = image_display.accepted_frames

            # get the dictionary of accepted frames for each entity
            entity_accepted_frames = image_display.accepted_frames_per_entity

            # gets the indexes of the accepted frames which is what the function
            # 'get_labeled_accepted_frames_and_embeddings' is expecting
            accepted_frames_list_input = [
                i
                for i, frame_accepted in enumerate(
                    image_display.accepted_frames
                )
                if frame_accepted
            ]

            # concatenation of batches happens in 'get_labeled_accepted_frames_and_embeddings'
            """NOTE: this function has been updated so now the dataframe will have a column \
                for each entity's label"""
            labeled_df = (
                # top_k_frames is always updated at the end of the while loop,
                # on the first iteration it's just VIVA's results. On the second iteration
                # it's VIVA's results + the results of the first iteration
                self._similarity_search_obj.get_labeled_accepted_frames_and_embeddings(
                    accepted_frames_list_input,
                    top_k_frames,
                    entity_accepted_frames,
                )
            )

            """
            NOTE:get the standard embedding list
            
            It looks like this:
            [(embedding, label)_1, (embedding, label)_2, (embedding, label)_3...(embedding, label)_n]
            """
            standard_embedding_list_SalsaSVM = (
                self._embedding_svm.get_standard_embedding_list(labeled_df)
            )
            (
                X_train,
                X_test,
                Y_train,
                Y_test,
            ) = self._embedding_svm.preprocess_embedding_data(
                standard_embedding_list_SalsaSVM
            )

            salsaSVM_train_start = now()
            self._svm_obj._model = self._svm_obj.train_svm(X_train, Y_train)
            salsaSVM_train_end = now()

            print(
                f"Time to train svm on batch number {self._similarity_search_obj.batch_num} is\
                    {salsaSVM_train_end - salsaSVM_train_start} seconds"
            )
            total_time_to_train_SalsaSVM += (
                salsaSVM_train_start - salsaSVM_train_end
            )

            self._svm_obj._recall = self._svm_obj.get_recall(X_test, Y_test)
            print(
                f"svm recall on batch {self._similarity_search_obj.batch_num} is {self._svm_obj._recall}"
            )

            # run similarity search again and get new accepted frames ->
            # at this point it's getting the mean of all the accepted frames,
            # not just the batch
            mean = self._similarity_search_obj.get_mean_embedding(labeled_df)

            # after the first iteration this needs to be passed into the
            # 'get_labeled_accepted_frames_and_embeddings' function
            # the batch_num is being updated inside this function
            top_k_frames = self._similarity_search_obj.knn_similiarity_search(
                mean, labeled_df, all_images_df
            )

            if self._svm_obj._recall > best_recall:
                print(
                    f"Recall improved from {best_recall} to \
                        {self._svm_obj._recall}"
                )
                best_recall = self._svm_obj._recall
                # only save the svm model if there is an improvement
                self._svm_obj.save_model(
                    self._svm_obj, self._similarity_search_obj.prompt
                )

            self._chip_manager.train_chips(labeled_df)

        if self._svm_obj._recall > self._model_threshold:
            self.salsa_plus_chips(total_time_to_train_SalsaSVM, all_images_df)

        else:
            self.chips_pipeline(all_images_df)

    def salsa_plus_chips(self, total_time_to_train_SalsaSVM, all_images_df):
        """
        This function is called when the SalsaSVM has successfully finished
        and the Salsa._svm_obj has a high enough recall such that we can
        use it to filter out frames.

        Args:
            total_time_to_train_SalsaSVM (timeit obj): This is used to keep
            track of the number of seconds that it took to train the general
            SalsaSVM (i.e. Salsa._svm_obj)

            all_images_df (dataframe): This is the dataframe that contains all
                                       of the frames in the video.
        """
        # SALSA SVM has successfully finished training with the desired recall
        print(f"Total seconds to train svm: {total_time_to_train_SalsaSVM}")

        all_images_df.loc[:, "embedding_array"] = all_images_df[
            "clip_embedding"
        ].apply(lambda x: self._similarity_search_obj._load_embedding(x))

        # initialize an empty list to store predicted labels
        batch_size = int(len(all_images_df) * 0.20)
        predicted_labels = []

        # process the data in batches
        for i in range(0, len(all_images_df), batch_size):
            print(f"we're on batch number { i / batch_size }")
            batch = np.vstack(
                all_images_df.iloc[i : i + batch_size]["embedding_array"]
            )
            batch_labels = self._svm_obj._model.predict(batch)
            predicted_labels.extend(batch_labels)

        print("Finished predicting labels")
        all_images_df["predicted_label"] = predicted_labels
        print("Finished adding predicted labels to dataframe")

        # get dataframe with all images where the predicted label is 1
        df_to_write_out = all_images_df[all_images_df["predicted_label"] == 1]
        print("Finished filtering dataframe")

        print("Fine-tuning with chips")
        fine_tuned_df_to_write_out = self._chip_manager.apply_chips(
            df_to_write_out
        )

        # NOTE: this is a bottleneck
        # Remove duplicates
        # preds_ids, preds = get_mmr_preds(df_to_write_out, MMR_ALPHA)
        # print("Finished getting MMR preds")
        # preds_pd = filter_by_similarity(df_to_write_out, preds_ids, preds)
        # print("Finished filtering by similarity")

        # NOTE: temporary solution: the default number of frames to jump is \
        # set to 10
        fine_tuned_df_to_write_out = self._naive_filter(
            fine_tuned_df_to_write_out
        )

        print(f"Writing out {len(fine_tuned_df_to_write_out)} frames")
        self._similarity_search_obj.SVMImageIO.write_out_frames(
            fine_tuned_df_to_write_out, svm_output=True
        )

    def chips_pipeline(self, all_images_df):
        """
        This function is called when the general SalsaSVM has not successfully
        passed the required threshold. This is generally the case when trying
        to train one specialed SVM on the semantically desired class or
        superclass is not enough. So chips_pipeline leverages the chips that
        the user is able to define and also that Salsa is able to automatically
        deduce from the prompt to train specialized SVM's on a class by class
        basis and then use those SVM's to filter out frames.

        Args:
            all_images_df (dataframe): dataframe of all the frames in the video
        """
        all_images_df_chips = self._chip_manager.apply_chips(all_images_df)
        df_to_write_out = self._naive_filter(all_images_df_chips)
        print(f"Writing out {len(df_to_write_out)} frames")
        self._similarity_search_obj.SVMImageIO.write_out_frames(
            df_to_write_out, chips_pipeline=True
        )

    def show_me_more(self):
        """
        This is an exposed function that allows the user to get more
        frames based on their preferences in mode and method.
        """
        if self._mode == "exploitation" and self._method == "embeddings":
            self._show_me_more_embedding()
