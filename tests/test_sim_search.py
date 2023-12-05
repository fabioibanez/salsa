import logging
import sys
import os
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")

sys.path.append("/opt/viva/analysis")
sys.path.append("/opt/viva/analysis/salsa")
from image_display import ImageDisplay
from similarity_search_class import SimilaritySearch
from similarity_search_class import (
    get_accepted_frames_bool_list,
    write_out_frames_at_index,
)

base_analysis_dir = "analysis"
base_labeled_df_dir = os.path.join(base_analysis_dir, "salsa/labeled_dfs")
base_models_dir = os.path.join(base_analysis_dir, "salsa/models")
base_output_dir = "output"
base_results_dir = os.path.join(base_analysis_dir, "results")
base_embeddings_dir = os.path.join(base_output_dir, "embeddings")
clip_embedding_cache = {}  # {Key: filename, Value: embedding}


def test_sim_search_iter0():
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


def test_sim_search_n_batches(dataset: str, prompt: str):
    similarity_search_obj = SimilaritySearch(prompt, dataset)
    # initialize state
    accepted_frames_list_input = []

    # load in without duplicates!
    all_images_df = pd.read_parquet(
        os.path.join(
            base_results_dir,
            f"results_variable,{dataset},clc@0.1,{prompt}-otherstuff",
        )
    )
    # removes all duplicates
    all_images_df = all_images_df.drop_duplicates(subset=["local_id"])
    all_images_df = all_images_df.reset_index()

    # on iter0 Salsa uses the results from the video analytics platform
    top_k_frames = pd.read_parquet(
        os.path.join(
            base_results_dir,
            f"similarity_variable,{dataset},clc@0.1,{prompt}-otherstuff_tiling",
        )
    )

    while True:
        # Empty entity list
        image_display = ImageDisplay(top_k_frames, [])
        image_display.start()

        # boolean list of accepted frames for this batch; updates on a batch-basis
        accepted_frames_list_input = image_display.accepted_frames

        # get the dictionary of accepted frames for each entity
        entity_accepted_frames = image_display.accepted_frames_per_entity

        # gets the indexes of the accepted frames which is what the function
        # 'get_labeled_accepted_frames_and_embeddings' is expecting
        accepted_frames_list_input = [
            i
            for i, frame_accepted in enumerate(image_display.accepted_frames)
            if frame_accepted
        ]

        # concatenation of batches happens in 'get_labeled_accepted_frames_and_embeddings'
        """NOTE: this function has been updated so now the dataframe will have a column \
            for each entity's label"""
        labeled_df = (
            # top_k_frames is always updated at the end of the while loop,
            # on the first iteration it's just VIVA's results. On the second iteration
            # it's VIVA's results + the results of the first iteration
            similarity_search_obj.get_labeled_accepted_frames_and_embeddings(
                accepted_frames_list_input,
                top_k_frames,
                entity_accepted_frames,
            )
        )
        # run similarity search again and get new accepted frames ->
        # at this point it's getting the mean of all the accepted frames,
        # not just the batch
        mean = similarity_search_obj.get_mean_embedding(labeled_df)

        # after the first iteration this needs to be passed into the
        # 'get_labeled_accepted_frames_and_embeddings' function
        # the batch_num is being updated inside this function
        top_k_frames = similarity_search_obj.knn_similiarity_search(
            mean, labeled_df, all_images_df
        )


if __name__ == "__main__":
    test_sim_search_n_batches("ucf101", "drumming")
