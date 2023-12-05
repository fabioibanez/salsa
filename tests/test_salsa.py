import sys

sys.path.append("/opt/viva/analysis/salsa")
from salsa import Salsa


# NOTE: all tests done are in exploitation mode
def test_init():
    salsa_obj = Salsa(
        "crashes", "fastandfurious", "embeddings", "exploitation"
    )
    assert salsa_obj._method == "embeddings"
    assert salsa_obj._mode == "exploitation"
    assert len(salsa_obj.entities) == 1


def test_chip_person_crossing_the_street():
    # test the new input method
    salsa_obj = Salsa(
        "person_crossing_the_street",
        "bdd",
        "embeddings",
        "exploitation",
    )
    salsa_obj.show_me_more()


def test_chip_crashes():
    salsa_obj = Salsa(
        "crashes", "fastandfurious", "embeddings", "exploitation"
    )
    salsa_obj.show_me_more()


def test_chip_interview():
    salsa_obj = Salsa("interview", "tvnews", "embeddings", "exploitation")
    salsa_obj.show_me_more()


def test_chip_drumming():
    salsa_obj = Salsa("drumming", "ucf101", "embeddings", "exploitation")
    salsa_obj.show_me_more()


if __name__ == "__main__":
    # test_chip_crashes()
    test_chip_drumming()


# def main():
#     # TODO: modify the salsa prompt such that the spaces are '_'
#     # salsa_person_crossing_the_street_obj = Salsa("personcrossingthestreet", "bdd", "embeddings", "exploitation")
#     # salsa_person_crossing_the_street_obj.show_me_more()
#     # salsa_person_crossing_the_street_obj = Salsa("personcrossingthestreet", "bdd", "embeddings", "exploitation")

#     ucf101_frames_df_q1 = pd.read_parquet(
#         "/opt/viva/analysis/results/results_variable,ucf101,clc@0.1,playingpiano-otherstuff"
#     )
#     ucf101_frames_df_q2 = pd.read_parquet(
#         "/opt/viva/analysis/results/results_variable,ucf101,clc@0.1,jumpingjacks-otherstuff"
#     )

#     print(ucf101_frames_df_q1["label"])
