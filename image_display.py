import numpy as np
import matplotlib.pyplot as plt


class ImageDisplay:
    def __init__(self, dataframe, entities):
        self.dataframe = dataframe
        self.entities = entities
        self.state = "viewing"
        # this is for the general SalsaSVM that we are training
        self.accepted_frames = []
        self.accepted_frames_per_entity = {entity: [] for entity in entities}
        self.current_entity_labels = {entity: False for entity in entities}

    def onclick(self, event):
        if self.state == "viewing":
            print("IN VIEWING MODE")
            if event.key == "y":
                self.accepted_frames.append(True)  # General SVM
                self.state = "labeling"

            elif event.key == "n":
                self.accepted_frames.append(False)  # General SVM
                self.state = "labeling"
                # refer to line 61 this might not be necessary
                self.current_entity_labels = {
                    entity: False for entity in self.entities
                }  # Reset labels

            elif event.key == "x":
                self.accepted_frames.append(False)  # General SVM
                for entity in self.entities:
                    self.accepted_frames_per_entity[entity].append(False)
                self.next_image()

            else:
                print(
                    "In viewing mode. Use 'y' for full acceptance \
                    and labeling,'n' if entities are present but \
                    not accepted, and 'x' for complete reject."
                )

        if self.state == "labeling":
            print("IN LABELING MODE")
            # this lets the user know which key to pick for each entity
            print("#####  KEY  ######")
            for i, entity in enumerate(self.entities):
                print("press", i + 1, "to label", entity)
            print("### END OF KEY ###")
            if event.key in [str(i + 1) for i, _ in enumerate(self.entities)]:
                entity = self.entities[int(event.key) - 1]
                self.current_entity_labels[entity] = True
                print(f"{entity} detected in Image {self.current_image}")
                for key, val in self.current_entity_labels.items():
                    if val:
                        print(f"{key} detected in Image {self.current_image}")
                    else:
                        print(
                            f"{key} not detected in Image {self.current_image}"
                        )

            elif event.key == "c":
                self.store_labeled_data()
                self.state = "viewing"  # Move back to viewing mode
                self.current_entity_labels = {
                    entity: False for entity in self.entities
                }  # Reset labels
                self.next_image()  # Move to next image after labeling entities

            else:
                print(
                    "In labeling mode. Use number keys for entities, and 'c' to continue."
                )

    def store_labeled_data(self):
        """
        stores the labeled data for entities
        """
        for entity, present in self.current_entity_labels.items():
            self.accepted_frames_per_entity[entity].append(present)

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
                break

            except Exception as e:
                print(
                    f"Failed to open image due to {e}. Moving on to the next one..."
                )

    # Might not need this given that I'm creating a new ImageDisplay object for each batch
    def reset_for_new_batch(self):
        self.accepted_frames_per_entity = {
            entity: [] for entity in self.entities
        }
        self.current_entity_labels = {
            entity: False for entity in self.entities
        }

    def start(self):
        self.image_iterator = iter(self.dataframe["frameuri"])
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self.onclick)
        self.next_image()
