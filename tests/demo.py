import os
import sys
import matplotlib
from timeit import default_timer as now
import numpy as np
sys.path.append("/opt/viva/analysis/salsa")
from salsa import Salsa
import numpy as np
import cv2
import os
from tkinter import Tk, Label, Frame, Scrollbar, Canvas
from PIL import Image, ImageTk
import numpy as np
matplotlib.use("TkAgg")  # Use the TkAgg backend to handle image display

def display_images(folder_path):
    root = Tk()
    
    # Create a frame to hold the canvas and scrollbar
    frame = Frame(root)
    frame.pack(fill='both', expand='yes')
    
    # Create a canvas for the images
    canvas = Canvas(frame, bg='white')
    canvas.pack(side='left', fill='both', expand='yes')
    
    # Create a vertical scrollbar
    scrollbar = Scrollbar(frame, orient='vertical', command=canvas.yview)
    scrollbar.pack(side='right', fill='y')
    
    # Attach the scrollbar to the canvas
    canvas.config(yscrollcommand=scrollbar.set)
    
    # Create a frame inside the canvas to hold the images
    image_frame = Frame(canvas)
    canvas.create_window((0,0), window=image_frame, anchor='nw')
    
    row, col = 0, 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            try:
                im = Image.open(image_path)
                img = ImageTk.PhotoImage(im)
                label = Label(image_frame, image=img)
                label.image = img  # Keep a reference to prevent garbage collection
                label.grid(row=row, column=col)
                
                col += 1
                if col >= 5:  # Number of columns
                    col = 0
                    row += 1
            except Exception as e:
                print(f"Failed to open image due to {e}. Moving on to the next one...")
                
    image_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox('all'))
    
    root.mainloop()



def calculate_similarity(uri1, uri2):
    img1 = np.load(uri1)
    img2 = np.load(uri2)
    if img1.shape != img2.shape:
        # make the same shape
        img2 = cv2.resize(img2, img1.shape)
    total_pixels = img1.shape[0] * img1.shape[1]
    
    similar_pixels = np.sum(img1 == img2)
    similarity_percentage = similar_pixels / total_pixels * 100
    return similarity_percentage > 30

def demo_write_outs(df):
    """
    This function writes out the frames produced during the demo
    in case they're of any interest

    Args:
        df (dataframe): contains the frames that svm predicts meet
        the user's criteria/standard.
    """
    for index, row in df.iterrows():
        frame = row["frameuri"]
        nparr = np.load(frame)
        frame = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)
        # imwrite out frames to folder, if it doesn't exist then create it
        if not os.path.exists("tmp/frames/demo"):
            os.makedirs("tmp/frames/demo")

        cv2.imwrite(
            f"tmp/frames/demo/_svm_demo_{index}.jpg",
            frame,
        )


def streamlined_query(self, accepted_frame_indexes, num_batches):
    """
    Streamlined function for simple demoing purposes
    `Extends the Salsa class with this demo function
    """
    # initialize state
    best_recall = 0
    
    # keep track of batch number
    batch_num = 0

    # load in without duplicates!
    all_images_df = self._similarity_search_obj.SVMImageIO.get_all_images_df()

    # on iter0 Salsa uses the results from the video analytics platform
    top_k_frames = self._similarity_search_obj.SVMImageIO.get_parquet(
        self._similarity_search_obj.SVMImageIO.results_pq
    )
    total_time_to_train_SalsaSVM = 0
    while (
        batch_num < num_batches
    ):
        
        # gets the indexes of the accepted frames which is what the function
        # 'get_labeled_accepted_frames_and_embeddings' is expecting
        accepted_frames_list_input = accepted_frame_indexes[f"batch{batch_num}"]
        
        

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
            )
        )
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
            f"Time to train svm on batch number {batch_num} is\
            {salsaSVM_train_end - salsaSVM_train_start} seconds"
        )
        total_time_to_train_SalsaSVM += (
            salsaSVM_train_end - salsaSVM_train_start
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
        batch_num += 1

    print(f"The total time to train SalsaSVM is {total_time_to_train_SalsaSVM}")
    if self._svm_obj._recall > self._model_threshold:
        all_images_df.loc[:, "embedding_array"] = all_images_df[
            "clip_embedding"
        ].apply(lambda x: self._similarity_search_obj._load_embedding(x))
        
        # initialize an empty list to store predicted labels
        batch_size = int(len(all_images_df) * 0.20)
        predicted_labels = []

        # process the data in batches
        predicting_labels_start = now()
        for i in range(0, len(all_images_df), batch_size):
            print(f"we're on batch number { i / batch_size }")
            batch = np.vstack(
                all_images_df.iloc[i : i + batch_size]["embedding_array"]
            )
            batch_labels = self._svm_obj._model.predict(batch)
            predicted_labels.extend(batch_labels)
        predicting_labels_end = now()
        print("Finished predicting labels")
        
        applying_labels_start = now()
        all_images_df["predicted_label"] = predicted_labels
        applying_labels_end = now()
        print("Finished adding predicted labels to dataframe")

        # get dataframe with all images where the predicted label is 1
        df_to_write_out = all_images_df[all_images_df["predicted_label"] == 1]
        print("Finished filtering dataframe")
       
        print("STATISTICS")
        print("===========")
        print(f"Total time to train SalsaSVM: {total_time_to_train_SalsaSVM}")
        print(f"Total time to predict labels: {predicting_labels_end - predicting_labels_start}")
        print(f"Total time to apply labels: {applying_labels_end - applying_labels_start}")
        print("===========")

        # remove duplicates
        to_remove = []
        for i in range(len(df_to_write_out) - 1):
            uri1 = df_to_write_out.iloc[i]["frameuri"]
            uri2 = df_to_write_out.iloc[i + 1]["frameuri"]
            
            if calculate_similarity(uri1, uri2):
                to_remove.append(i)
        # drop dupes
        df_to_write_out = df_to_write_out.reset_index()
        df_to_write_out = df_to_write_out.drop(to_remove)
        
        return df_to_write_out


Salsa.demo = streamlined_query


def streamlined_drumming():
    accepted_frames_per_batch = {
        "batch0": [0, 10, 14],
        "batch1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    }
    num_batches = 2
    salsa_obj = Salsa("drumming", "ucf101", "embeddings", "exploitation")
    df = salsa_obj.demo(accepted_frames_per_batch, num_batches)
    # Dummy list of image paths for demonstration purposes
    # image_paths = df["frameuri"].tolist()
    display_images("/opt/viva/tmp/frames/demo")
    # demo_write_outs(df)


def main():
    streamlined_drumming()


if __name__ == "__main__":
    main()