import os
import logging
import sklearn as sk
from sklearn.cluster import KMeans
import numpy as np
import cv2
import torch
import clip
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# CLIP import stuff
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

base_analysis_dir = "analysis"
base_labeled_df_dir = os.path.join(base_analysis_dir, "salsa/labeled_dfs")
base_models_dir = os.path.join(base_analysis_dir, "salsa/models")
base_output_dir = "output"
base_results_dir = os.path.join(base_analysis_dir, "results")
base_embeddings_dir = os.path.join(base_output_dir, "embeddings")
clip_embedding_cache = {}  # {Key: filename, Value: embedding}

logging.basicConfig(
    # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.path.join(base_analysis_dir, "logs/app.log"),
    filemode="w",  # Set the file mode (e.g., 'w' for write, 'a' for append)
)


# TESTING FUNCTION
def get_accepted_frames_bool_list(accepted_frame_list: list, num_frames: int):
    """
    This function makes it easier for a user to get a list of booleans 
    representing the accepted frames

    Args:
        accepted_frame_list (list): a list of indexes of the accepted frames,
        starting at 1
        num_frames (int): the total number of frames in the list, so theoretically
        len(accepted_frames_list) == num_frames

    Returns:
        list: a list of booleans where the indexes of the accepted frames are\
            True and the rest are False
    """
    return_list = [False] * num_frames
    try:
        for idx in accepted_frame_list:
            return_list[idx - 1] = True
    except Exception as e:
        logging.error(
            "ERROR: return_list[idx] = True is failing, it's probably that\
                your indexes are messed up!",
            e,
        )
    return return_list


# TESTING FUNCTION
def write_out_frames_at_index(accepted_frames_list, df_with_frames):
    for index, row in df_with_frames.iterrows():
        if accepted_frames_list[index]:
            frame = row["frameuri"]
            nparr = np.load(frame)
            frame = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)
            if not os.path.exists(f"tmp/frames/testing_index"):
                os.makedirs(f"tmp/frames/testing_index")
            cv2.imwrite(f"tmp/frames/testing_index/{index}.jpg", frame)


class SVMImageIO:
    def __init__(self, prompt: str, dataset: str):
        self.prompt = prompt
        self.dataset = dataset

        self.all_frames_pq = os.path.join(
            base_results_dir,
            f"results_variable,{self.dataset},clc@0.1,{self.prompt}-otherstuff",
        )
        self.results_pq = os.path.join(
            base_results_dir,
            f"similarity_variable,{self.dataset},clc@0.1,{self.prompt}-otherstuff_tiling",
        )

    def get_parquet(self, parquet):
        """
        This function reads in a parquet file and returns a corresponding dataframe

        Args:
            parquet (str): a path to the folder of parquet files

        Returns:
            dataframe: returns a dataframe of the contents in the parquet files
        """
        return pd.read_parquet(parquet)

    def get_all_images_df(self):
        all_images_df = self.get_parquet(
            os.path.join(
                base_results_dir,
                f"results_variable,{self.dataset},clc@0.1,{self.prompt}-otherstuff",
            )
        )
        # removes all duplicates
        all_images_df = all_images_df.drop_duplicates(subset=["local_id"])
        return all_images_df

    def write_out_frames(
        self,
        df_with_frames,
        svm_output=False,
        cosine_test=False,
        knn_test=False,
        chips_pipeline=False,
    ):
        """
        This function writes out the frames to a folder called "tmp/frames/{dataset}"
        and it is intended to be called on the dataframe containing the top-k results

        Args:
            df_with_frames (dataframe): dataframe with 'frameuri' column, typically
            the dataframe with the top-k results but can be any dataframe with a valid
            'frameuri' column
        """
        for index, row in df_with_frames.iterrows():
            frame = row["frameuri"]
            nparr = np.load(frame)
            frame = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)
            # imwrite out frames to folder, if it doesn't exist then create the folder
            if not os.path.exists(f"tmp/frames/{self.dataset}"):
                os.makedirs(f"tmp/frames/{self.dataset}")
            if svm_output:
                # if svm_output is true, then we want to make sure we know what frames the svm returned so we name them accordingly
                cv2.imwrite(
                    f"tmp/frames/{self.dataset}/{self.prompt}_svm_benchmark_{index}.jpg",
                    frame,
                )
                continue
            if cosine_test:
                cv2.imwrite(
                    f"tmp/frames/{self.dataset}/{self.prompt}_cosine_{index}.jpg",
                    frame,
                )
                continue
            if knn_test:
                cv2.imwrite(
                    f"tmp/frames/{self.dataset}/{self.prompt}_knn_{index}.jpg",
                    frame,
                )
                continue
            if chips_pipeline:
                cv2.imwrite(
                    f"tmp/frames/{self.dataset}/{self.prompt}_chips_{index}.jpg",
                    frame,
                )
                continue
            cv2.imwrite(
                f"tmp/frames/{self.dataset}/{self.prompt}_{index}.jpg",
                frame,
            )
        output_type = (
            "svm"
            if svm_output
            else "cosine"
            if cosine_test
            else "knn"
            if knn_test
            else "chips"
            if chips_pipeline
            else ""
        )
        print(
            f"frames written out to: tmp/frames/{self.dataset}/***_{output_type.split('_')[0]}_***.jpg"
        )


class SimilaritySearch:
    def __init__(self, prompt: str, dataset: str):
        self.prompt = prompt
        self.dataset = dataset
        # want to update this so that we can save the batches of the results, which will be distinct
        self.batch_num = 1
        self.SVMImageIO = SVMImageIO(prompt, dataset)
        self.clip_embedding_cache = {}

    def _load_embedding(self, clip_embedding_path):
        """_summary_

        Args:
            clip_embedding_path (str): in the form of "filename.npy#index"
        Returns:
            row (np array): the embedding associated with the path that is passed in
        """
        filename, index = clip_embedding_path.split("#")
        index = int(index)

        # Load Numpy array from file or cache
        if filename not in self.clip_embedding_cache:
            # adds the embedding to the cache
            arr = np.load(filename)
            self.clip_embedding_cache[filename] = arr
        else:
            arr = self.clip_embedding_cache[filename]

        # Extract row vector
        row = arr[index]

        return row

    def get_mean_embedding(self, dataframe_with_embeddings):
        """
        This function finds the mean of the embeddings where the "svm_label" is 1

        Args:
            dataframe_with_embeddings (dataframe): a dataframe with all the returned
            images and their embeddings

        Returns:
            mean (int): mean of the embeddings of the accepted frames
        """
        embeddings = dataframe_with_embeddings[
            dataframe_with_embeddings["svm_label"] == 1
        ]
        e_a = embeddings["embedding_array"]
        stacked_ea = np.stack((e_a.values))
        mean = np.mean(stacked_ea, axis=0)
        return mean

    def get_labeled_accepted_frames_and_embeddings(
        self,
        accepted_frames_lst: list,
        top_k_dataframe,
        entity_accepted_frames=None,
    ):
        """
        This function takes in an accepted_frames_lst and returns a dataframe
        where the frames at the true indexes in the accepted frames list are
        labeled as 1 and the rest are labeled as 0. It also saves this dataframe
        in the base_labeled_df_dir folder. This is because all of these
        labeled dfs are going to be necessary in order to train the SVM.

        This function is called after the 'get_accepted_frames_bool_list' function

        Args:
            accepted_frames_lst (list): a list of indexes of the accepted frames

        Returns:
            parquet_df (dataframe): A dataframe that contains the embeddings of the
            accepted frames and the rejected frames,
            i.e. the dataframe is effectively labeled.
        """
        parquet_folder_of_results = f"similarity_variable,{self.dataset},clc@0.1,{self.prompt}-otherstuff_tiling"

        # this is only 30 frames
        parquet_df = top_k_dataframe

        # TODO: Delete this, for testing purposes
        if "level_0" in parquet_df.columns:
            parquet_df = parquet_df.drop(columns=["level_0"])
        parquet_df = parquet_df.reset_index()

        # Label the frames as accepted or rejected, checks the positional index
        parquet_df["svm_label"] = parquet_df.index.map(
            lambda x: 1 if x in accepted_frames_lst else 0
        )

        # Label the frames as accepted or rejected for each entity
        if entity_accepted_frames is not None:
            for entity in entity_accepted_frames:
                accepted_frames_lst_entity = [
                    i
                    for i, frame_accepted in enumerate(
                        entity_accepted_frames[entity]
                    )
                    if frame_accepted
                ]

            parquet_df[f"{entity}_svm_label"] = parquet_df.index.map(
                lambda x: 1 if x in accepted_frames_lst_entity else 0
            )

        parquet_df.loc[:, "embedding_array"] = parquet_df[
            "clip_embedding"
        ].apply(lambda x: self._load_embedding(x))

        # save the labeled dataframe --> need this for batch training of SVM
        (
            type,
            dataset,
            model,
            prompt_and_otherstuff,
        ) = parquet_folder_of_results.split(",")
        prompt, otherstuff = prompt_and_otherstuff.split("-")

        # makes sure to batch dataframes so that we can train the SVM on all the data
        if self.batch_num > 1:
            superbatch = self.SVMImageIO.get_parquet(
                os.path.join(
                    base_labeled_df_dir,
                    f"labeled_df_{dataset}_{prompt}.parquet",
                )
            )
            # concatenate superbatch and parquet_df
            parquet_df = pd.concat([superbatch, parquet_df])

        # writes to the labeled_df folder
        parquet_df.to_parquet(
            os.path.join(
                base_labeled_df_dir,
                f"labeled_df_{dataset}_{prompt}.parquet",
            )
        )

        # loads the embeddings of both accepted and rejected frames which is good for our SVM
        return parquet_df

    def knn_similiarity_search(
        self,
        mean: int,
        dataframe_with_embeddings,
        all_images_df=None,
        k=30,
    ):
        """
        This function takes in a mean embedding and returns the top k images that
        are most similar to the mean embedding

        Args:
            mean (int): mean of the embeddings of the accepted frames

            dataframe_with_embeddings (dataframe): a dataframe with all
            the returned images and their embeddings

            all_images_df (dataframe): a dataframe of all the images in the dataset
            k (int, optional): Top-k images k defaults to 30.

        Returns:
            top_k_images (dataframe): a dataframe with the top-k most similiar images by embeddings
        """
        if all_images_df is None:
            all_images_df = self.SVMImageIO.get_parquet(
                os.path.join(
                    base_results_dir,
                    f"results_variable,{self.dataset},clc@0.1,{self.prompt}-otherstuff",
                )
            )
        all_images_df = all_images_df.drop_duplicates(subset=["local_id"])
        all_images_df = all_images_df.reset_index()

        # make sure to get rid of images that we've already seen (i.e. the ones in the `dataframe_with_embeddings` df)
        all_images_df = all_images_df[
            ~all_images_df["local_id"].isin(
                dataframe_with_embeddings["local_id"]
            )
        ].copy()

        all_images_df.loc[:, "embedding_array"] = all_images_df[
            "clip_embedding"
        ].apply(lambda x: self._load_embedding(x))

        # stack all emebddings into a matrix
        embeddings_matrix = np.stack(all_images_df["embedding_array"].values)

        # create NearestNeighbors model
        # instead of 5 nearest I'm doing 30 so that we can deal with diversity later
        sklearn_model = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
        sklearn_model.fit(embeddings_matrix)

        distances, indices = sklearn_model.kneighbors([mean])

        top_k_images = all_images_df.iloc[indices[0]]

        self.batch_num += 1

        return top_k_images

    def cosine_similarity_search(
        self,
        mean: int,
        dataframe_with_embeddings,
        all_images_df=None,
        k=30,
    ):
        """
        This function takes in a mean embedding and returns the top k images
        that are most similar to the mean embedding

        Args:
            mean (array): mean of the embeddings of the accepted frames

            dataframe_with_embeddings (dataframe): a dataframe with all the
            returned images and their embeddings

            all_images_df (dataframe): a dataframe of all the images in the dataset
            k (int, optional): Top-k images k defaults to 30.

        Returns:
            top_k_images (dataframe): a dataframe with the top-k most similiar images by embeddings
        """
        if all_images_df is None:
            all_images_df = self.SVMImageIO.get_parquet(
                os.path.join(
                    base_results_dir,
                    f"results_variable,{self.dataset},clc@0.1,{self.prompt}-otherstuff",
                )
            )
        all_images_df = all_images_df.drop_duplicates(subset=["local_id"])
        all_images_df = all_images_df.reset_index()

        # make sure to get rid of images that we've already seen (i.e. the ones in the `dataframe_with_embeddings` df)
        all_images_df = all_images_df[
            ~all_images_df["local_id"].isin(
                dataframe_with_embeddings["local_id"]
            )
        ].copy()

        all_images_df.loc[:, "embedding_array"] = all_images_df[
            "clip_embedding"
        ].apply(lambda x: self._load_embedding(x))

        # stack all emebddings into a matrix
        embeddings_matrix = np.stack(all_images_df["embedding_array"].values)

        # create NearestNeighbors model
        # instead of 5 nearest I'm doing 30 so that we can deal with diversity later
        sklearn_model = NearestNeighbors(
            n_neighbors=k, algorithm="brute", metric="cosine"
        )
        sklearn_model.fit(embeddings_matrix)

        distances, indices = sklearn_model.kneighbors([mean])

        top_k_images = all_images_df.iloc[indices[0]]

        self.batch_num += 1

        return top_k_images

    def test_knn_similarity_search(self, accepted_frames_list_input: list):
        """
        This function is used to test the knn similarity search functions.
        It will return the top-k images and write them out to a folder called
        "tmp/frames/{dataset}"

        Args:
            accepted_frames_list_input (list): a list of indexes of the accepted frames,
            starting at 1
        """
        viva_top_k_frames = self.SVMImageIO.get_parquet(
            self.SVMImageIO.results_pq
        )
        # in prod we always explicitly pass in the top_k_frames, we must do the same here
        labeled_df = self.get_labeled_accepted_frames_and_embeddings(
            accepted_frames_list_input, viva_top_k_frames
        )
        # get the mean embedding
        mean = self.get_mean_embedding(labeled_df)
        top_k_images = self.knn_similiarity_search(mean, labeled_df)
        self.SVMImageIO.write_out_frames(top_k_images, knn_test=True)

    def test_cosine_similarity_search(self, accepted_frames_list_input: list):
        viva_top_k_frames = self.SVMImageIO.get_parquet(
            self.SVMImageIO.results_pq
        )
        # in prod we always explicitly pass in the top_k_frames, we must do the same here
        labeled_df = self.get_labeled_accepted_frames_and_embeddings(
            accepted_frames_list_input, viva_top_k_frames
        )
        # get the mean embedding
        mean = self.get_mean_embedding(labeled_df)
        top_k_images = self.cosine_similarity_search(mean, labeled_df)
        self.SVMImageIO.write_out_frames(top_k_images, cosine_test=True)


def test():
    """
    This function is used to test the SimilaritySearch class, and also show why similarity search
    alone is not enough to good and fast results.

    The tests are as follows:
    1) TEST 1 :: test for crashes, more specifically crashes that involve a car on fire or an explosion.

    2) TEST 2 :: test for drumming, where we're only looking for drumming and not other instruments. We are
    also not looking for a drum set, but rather want a more traditional drum.

    3) TEST 3 :: test for playing piano, where we are interested in frames where a person is playing piano.

    4) TEST 4 :: test for jumpiing jacks, where we are interested in frames where a person is activey doing
    a jumping jack.
    """
    ######################## TEST 1 ########################
    logging.info(
        "INFO::Test 1: Testing that the correct frames for 'crashes' in 'fastandfurious' are being written out"
    )
    similarity_search_obj_fast_and_furious = SimilaritySearch(
        "crashes", "fastandfurious"
    )
    accepted_frames_list_input = get_accepted_frames_bool_list([6, 11, 26], 30)
    write_out_frames_at_index(
        accepted_frames_list_input,
        similarity_search_obj_fast_and_furious.SVMImageIO.get_parquet(
            similarity_search_obj_fast_and_furious.SVMImageIO.results_pq
        ),
    )

    logging.info(
        "INFO::Test 1: Running KNN Similarity Search on 'crashes' in 'fastandfurious'"
    )
    similarity_search_obj_fast_and_furious.test_knn_similarity_search(
        accepted_frames_list_input
    )

    logging.info(
        "INFO::Test 1: Running Cosine Similarity Search on 'crashes' in 'fastandfurious'"
    )
    similarity_search_obj_fast_and_furious.test_cosine_similarity_search(
        accepted_frames_list_input
    )

    ######################## TEST 2 ########################
    logging.info(
        "INFO::Test 2: Testing that the correct frames for 'drumming' in 'ucf101' are being written out"
    )
    similarity_search_obj_ucf_drumming = SimilaritySearch("drumming", "ucf101")
    accepted_frames_list_input = get_accepted_frames_bool_list([1, 11, 15], 30)
    write_out_frames_at_index(
        accepted_frames_list_input,
        similarity_search_obj_ucf_drumming.SVMImageIO.get_parquet(
            similarity_search_obj_ucf_drumming.SVMImageIO.results_pq
        ),
    )

    logging.info(
        "INFO::Test 2: Runing KNN Similarity Search on 'drumming' in 'ucf101'"
    )
    similarity_search_obj_ucf_drumming.test_knn_similarity_search(
        accepted_frames_list_input
    )

    logging.info(
        "INFO::Test 2: Runing Cosine Similarity Search on 'drumming' in 'ucf101'"
    )
    similarity_search_obj_ucf_drumming.test_cosine_similarity_search(
        accepted_frames_list_input
    )

    ######################## TEST 3 ########################
    logging.info(
        "INFO::Test 3: Testing that the correct frames for 'playing piano' in 'ucf101' are being written out"
    )
    similarity_search_obj_ucf_playing_piano = SimilaritySearch(
        "playingpiano", "ucf101"
    )
    accepted_frames_list_input = get_accepted_frames_bool_list([1], 30)
    write_out_frames_at_index(
        accepted_frames_list_input,
        similarity_search_obj_ucf_playing_piano.SVMImageIO.get_parquet(
            similarity_search_obj_ucf_playing_piano.SVMImageIO.results_pq
        ),
    )

    logging.info(
        "INFO::Test 3: Runing KNN Similarity Search on 'playing piano' in 'ucf101'"
    )
    similarity_search_obj_ucf_playing_piano.test_knn_similarity_search(
        accepted_frames_list_input
    )

    logging.info(
        "INFO::Test 3: Runing Cosine Similarity Search on 'playing piano' in 'ucf101'"
    )
    similarity_search_obj_ucf_playing_piano.test_cosine_similarity_search(
        accepted_frames_list_input
    )

    ######################## TEST 4 ########################
    logging.info(
        "INFO::Test 4: Testing that the correct frames for 'jumping jacks' in 'ucf101' are being written out"
    )
    similarity_search_obj_ucf_jumping_jacks = SimilaritySearch(
        "jumpingjacks", "ucf101"
    )
    accepted_frames_list_input = get_accepted_frames_bool_list([1, 4], 30)
    write_out_frames_at_index(
        accepted_frames_list_input,
        similarity_search_obj_ucf_jumping_jacks.SVMImageIO.get_parquet(
            similarity_search_obj_ucf_jumping_jacks.SVMImageIO.results_pq
        ),
    )

    logging.info(
        "INFO::Test 3: Runing KNN Similarity Search on 'jumping jacks' in 'ucf101'"
    )
    similarity_search_obj_ucf_jumping_jacks.test_knn_similarity_search(
        accepted_frames_list_input
    )

    logging.info(
        "INFO::Test 3: Runing Cosine Similarity Search on 'jumping jacks' in 'ucf101'"
    )
    similarity_search_obj_ucf_jumping_jacks.test_cosine_similarity_search(
        accepted_frames_list_input
    )


if __name__ == "__main__":
    test()
