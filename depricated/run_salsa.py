import matplotlib
import sys

sys.path.append("/opt/viva/analysis")

matplotlib.use("TkAgg")  # Use the TkAgg backend to handle image display
import argparse
import os
from svm_utils import SalsaSVM
from similarity_search_class import (
    SimilaritySearch,
    get_accepted_frames_bool_list,
    load_embedding,
)
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from timeit import default_timer as now
from compute_accuracy_retrieval_simple import *


base_analysis_dir = "analysis"
base_results_dir = os.path.join(base_analysis_dir, "results")
base_salsa_dir = os.path.join(base_analysis_dir, "salsa")
threshold = 0.8
num_frames = 30
MMR_ALPHA = 0.0


class ImageDisplay:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # returns a list of booleans that correspond to the question "Do you accept this frame?"
        self.accepted_frames = []

    def onclick(self, event):
        if event.key == "y":
            self.accepted_frames.append(True)
            print(f"Image {self.current_image} accepted")
        elif event.key == "n":
            self.accepted_frames.append(False)
            print(f"Image {self.current_image} rejected")
        self.next_image()

    def next_image(self):
        while True:
            try:
                self.current_image = next(self.image_iterator)
                array = np.load(self.current_image)

                plt.imshow(array)
                plt.show()
                break

            except StopIteration:
                plt.close("all")
                print("End of image display.")
                print("Accepted Frames: ", self.accepted_frames)
                break

            except Exception as e:
                print(
                    f"Failed to open image due to {e}. Moving on to the next one..."
                )

    def start(self):
        self.image_iterator = iter(self.dataframe["frameuri"])
        fig = plt.figure()
        fig.canvas.mpl_connect("key_press_event", self.onclick)
        self.next_image()


def run_salsa(prompt: str, dataset: str):
    # initialize SimilaritySearch object
    similarity_search = SimilaritySearch(prompt, dataset)
    # initialize svm object
    svm = SalsaSVM(dataset, prompt)
    # specify that my svm is an embeddings SVM
    embeddings_svm = svm.EmbeddingSalsaSVMDataProcessor()
    # svm recall starts at 0
    best_recall = 0
    accepted_frames_list_input = []
    all_images_df = similarity_search.ImageIO.get_all_images_df()

    top_k_frames = similarity_search.ImageIO.get_parquet(
        similarity_search.ImageIO.results_pq
    )

    # use similarity search to get the accepted frames then train svm
    total_time_to_train_svm = 0
    while svm.recall < threshold:
        image_display = ImageDisplay(top_k_frames)
        image_display.start()

        # boolean list of accepted frames for this batch; updates on a batch-basis
        accepted_frames_list_input = image_display.accepted_frames

        # gets the indexes of the accepted frames which is what the function 'get_labeled_accepted_frames_and_embeddings' is expecting
        accepted_frames_list_input = [
            i
            for i, frame_accepted in enumerate(image_display.accepted_frames)
            if frame_accepted
        ]

        # concatenation of batches happens in 'get_labeled_accepted_frames_and_embeddings'
        labeled_df = (
            # top_k_frames is always updated at the end of the while loop, on the first iteration it's just VIVA's results. On the second iteration it's VIVA's results + the results of the first iteration
            similarity_search.get_labeled_accepted_frames_and_embeddings(
                accepted_frames_list_input, top_k_frames
            )
        )

        # get the standard embedding list
        """
        NOTE: this looks like:
        [(embedding, label)_1, (embedding, label)_2, (embedding, label)_3...(embedding, label)_n]
        """
        standard_embedding_list = embeddings_svm.get_standard_embedding_list(
            labeled_df
        )
        (
            X_train,
            X_test,
            Y_train,
            Y_test,
        ) = embeddings_svm.preprocess_embedding_data(standard_embedding_list)

        train_start = now()
        svm.model = svm.train_svm(X_train, Y_train)
        train_end = now()

        print(
            f"Time to train svm on batch number {similarity_search.batch_num} is {train_end - train_start} seconds"
        )
        total_time_to_train_svm += train_end - train_start

        svm.recall = svm.get_recall(X_test, Y_test)
        print(
            f"svm recall on batch {similarity_search.batch_num} is {svm.recall}"
        )

        # run similarity search again and get new accepted frames -> at this point it's getting the mean of all the accepted frames, not just the batch
        mean = similarity_search.get_mean_embedding(labeled_df)

        # after the first iteration this needs to be passed into the 'get_labeled_accepted_frames_and_embeddings' function
        # the batch_num is being updated inside this function
        top_k_frames = similarity_search.knn_similiarity_search(
            mean, labeled_df, all_images_df
        )

        if svm.recall > best_recall:
            print(f"Recall improved from {best_recall} to {svm.recall}")
            best_recall = svm.recall
        svm.save_model(svm.model, similarity_search.prompt)

    print(f"Total seconds to train svm: {total_time_to_train_svm}")

    # use the svm to get the accepted frames
    two_third_point = len(all_images_df) * 2 // 3

    all_images_df = all_images_df.iloc[two_third_point:]

    all_images_df.loc[:, "embedding_array"] = all_images_df[
        "clip_embedding"
    ].apply(lambda x: load_embedding(x))

    # TODO: have to do something about this, can make this dynamic?
    batch_size = int(len(all_images_df) * 0.20)
    print(f"the batch_size is {batch_size}")

    # initialize an empty list to store predicted labels
    predicted_labels = []

    # process the data in batches
    for i in range(0, len(all_images_df), batch_size):
        print(f"we're on batch number { i / batch_size }")
        batch = np.vstack(
            all_images_df.iloc[i : i + batch_size]["embedding_array"]
        )
        batch_labels = svm.model.predict(batch)
        predicted_labels.extend(batch_labels)

    all_images_df["predicted_label"] = predicted_labels

    # get dataframe with all images where the predicted label is 1
    df_to_write_out = all_images_df[all_images_df["predicted_label"] == 1]

    # Remove duplicates
    preds_ids, preds = get_mmr_preds(df_to_write_out, MMR_ALPHA)
    preds_pd = filter_by_similarity(df_to_write_out, preds_ids, preds)

    similarity_search.ImageIO.write_out_frames(preds_pd, svm_output=True)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run a query on a dataset")

    # Add the arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt for the query",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments as variables
    dataset = args.dataset
    prompt = args.prompt

    run_salsa(prompt, dataset)


if __name__ == "__main__":
    main()
