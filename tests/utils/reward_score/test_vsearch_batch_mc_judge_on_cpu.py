import asyncio

from verl.utils.reward_score.vsearch_batch import (
    _compute_rule_based_raw_final_answer_accuracy_v2,
    _parse_mc_options_from_question,
    _raw_final_answer_matches_mc_ground_truth,
    _raw_final_answer_matches_mc_ground_truth_v1,
    compute_accuracy_reward_with_final_answer_only,
)


def test_mc_judge_does_not_pass_rejected_gt_label():
    question = (
        "This image shows the front view of the ego car. What is the ego vehicle doing? "
        "(A) The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast. "
        "(B) The ego vehicle is turning left. "
        "(C) The ego vehicle is stopped. "
        "(D) The ego vehicle is reversing. "
        "(E) All the above answers are wrong."
    )
    answer = (
        "Based on the image, the road ahead curves to the right, but the vehicle is not driving very fast. "
        "Therefore option (A) is incorrect because it claims the vehicle is driving very fast. "
        "None of the provided options correctly describe both direction and speed, "
        "so the correct answer is (E) All the above answers are wrong."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="A",
            full_final_answer=answer,
        )
        is False
    )


def test_single_call_v1_preserves_legacy_rejected_gt_label_false_positive():
    question = (
        "This image shows the front view of the ego car. What is the ego vehicle doing? "
        "(A) The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast. "
        "(B) The ego vehicle is turning left. "
        "(C) The ego vehicle is stopped. "
        "(D) The ego vehicle is reversing. "
        "(E) All the above answers are wrong."
    )
    answer = (
        "Based on the image, the road ahead curves to the right, but the vehicle is not driving very fast. "
        "Therefore option (A) is incorrect because it claims the vehicle is driving very fast. "
        "None of the provided options correctly describe both direction and speed, "
        "so the correct answer is (E) All the above answers are wrong."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth_v1(
            question=question,
            ground_truth="A",
            full_final_answer=answer,
        )
        is True
    )
    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="A",
            full_final_answer=answer,
        )
        is False
    )


def test_final_answer_only_rule_version_dispatch_preserves_v1_and_enables_v2():
    question = (
        "This image shows the front view of the ego car. What is the ego vehicle doing? "
        "(A) The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast. "
        "(B) The ego vehicle is turning left. "
        "(C) The ego vehicle is stopped. "
        "(D) The ego vehicle is reversing. "
        "(E) All the above answers are wrong."
    )
    answer = (
        "Option (A) is incorrect because it claims the vehicle is driving very fast. "
        "The correct answer is (E) All the above answers are wrong."
    )

    legacy_score = asyncio.run(
        compute_accuracy_reward_with_final_answer_only(
            "mmlite",
            question,
            answer,
            "A",
            judge_client=None,
            rule_version="v1",
        )
    )
    fixed_score = asyncio.run(
        compute_accuracy_reward_with_final_answer_only(
            "mmlite",
            question,
            answer,
            "A",
            judge_client=None,
            rule_version="v2",
        )
    )

    assert legacy_score == 1.0
    assert fixed_score == 0.0


def test_mc_judge_uses_final_explicit_selection_not_any_label_mention():
    question = (
        "Which team won the game? "
        "(A) TS. (B) Tenn. (C) OHIO ST. (D) NCAA. (E) The image does not provide sufficient information."
    )
    answer = "Options (A) and (B) are visible in the broadcast graphic, but the final answer is (C) OHIO ST."

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="C",
            full_final_answer=answer,
        )
        is True
    )


def test_mc_judge_matches_concise_option_text_answers():
    question = (
        "Which team won the game? "
        "(A) TS. (B) Tenn. (C) OHIO ST. (D) NCAA. (E) The image does not provide sufficient information."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="C",
            full_final_answer="OHIO ST.",
        )
        is True
    )
    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="C",
            full_final_answer="Tenn.",
        )
        is None
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="C",
            full_final_answer="Final answer: (B) Tenn.",
        )
        is False
    )


def test_mc_judge_does_not_treat_ocr_text_prefix_as_option_label():
    question = (
        "What's the plate number of the car? "
        "(A) B HH 30H (B) B HH 3QH (C) B HH 3OH (D) B NH 30H (E) This image doesn't feature the plate number."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="A",
            full_final_answer="B HH 30H",
        )
        is True
    )


def test_mc_judge_parses_letter_dot_options():
    question = (
        "Which port does HTTP usually listen on? "
        "A. localhost:5000 B. localhost:5001 Choose 'A' or 'B'."
    )

    assert _parse_mc_options_from_question(question) == {
        "A": "localhost:5000",
        "B": "localhost:5001 Choose 'A' or 'B'.",
    }
    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="A",
            full_final_answer="The correct answer is A. localhost:5000.",
        )
        is True
    )


def test_mc_judge_accepts_choice_cues_and_punctuated_tail_labels():
    question = "Which has higher value? (A). option A (B). option B Choose the letter name from A, B."

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="B",
            full_final_answer="The correct choice is (B).",
        )
        is True
    )
    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="B",
            full_final_answer="Based on the table, option B is higher. (B). option B",
        )
        is True
    )


def test_mc_parser_does_not_parse_common_abbreviations_as_options():
    assert _parse_mc_options_from_question("When did Carol A. Tozzi, Ph.D. accept the assignment?") == {}
    assert _parse_mc_options_from_question("What is the D.C.No of the receipt?") == {}
    assert _parse_mc_options_from_question(
        "Where did Dominique Yantko receive their B.A. in Political Science?"
    ) == {}


def test_mc_judge_handles_rejection_option_without_label():
    question = (
        "What is the number of tricycles in the image? "
        "(A) 26 (B) 46 (C) 18 (D) 86 (E) The image does not feature tricycles."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="E",
            full_final_answer="The image does not feature any tricycles.",
        )
        is True
    )
    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="A",
            full_final_answer="The image does not feature any tricycles.",
        )
        is False
    )


def test_mc_judge_respects_non_e_zero_count_final_selection():
    question = (
        "What is the total number of bicycles and awning-tricycles in the image? "
        "(A) 16 (B) 9 (C) 12 (D) 0 (E) The image does not feature the objects"
    )
    answer = "No such vehicles are visible anywhere in the image. Therefore the total count is 0. (D) 0"

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="D",
            full_final_answer=answer,
        )
        is True
    )


def test_mc_judge_does_not_treat_take_none_action_as_rejection_option():
    question = (
        "What action should be taken? "
        "(A) The action is to keep going at the same speed. "
        "(B) The action is to take none, the reason is that there is no safety issue. "
        "(C) The action is to turn right. "
        "(D) The action is to accelerate. "
        "(E) All the above answers are wrong."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="B",
            full_final_answer="Take none, no safety issue",
        )
        is None
    )


def test_mc_judge_handles_markdown_bold_final_label():
    question = (
        "What is the status of the cars? "
        "(A) Many cars are moving and two are parked. "
        "(B) Two of the cars are moving, and three are parked. "
        "(C) Many cars are parked and three are moving. "
        "(D) Three of the cars are moving, and many are parked. "
        "(E) The image does not feature the object."
    )
    answer = (
        "Option (D) captures the overall scene accurately. "
        "Therefore, the correct answer is **(D) Three of the cars are moving, and many are parked.**"
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="D",
            full_final_answer=answer,
        )
        is True
    )


def test_mc_judge_ignores_rejected_later_option_mentions():
    question = (
        "What is the orientation of the truck in the image? "
        "(A) The right of the image "
        "(B) The bottom of the image "
        "(C) The left of the image "
        "(D) The top of the image "
        "(E) The image does not feature the truck"
    )
    answer = (
        "Therefore, the correct orientation of the truck is: **(C) The left of the image**. "
        "Option (B) is partially correct but less precise. "
        "The truck is clearly visible, so option (E) is incorrect."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="C",
            full_final_answer=answer,
        )
        is True
    )


def test_single_call_v2_defers_semantic_unanswerable_explanations_to_llm():
    assert (
        _compute_rule_based_raw_final_answer_accuracy_v2(
            question="What is the text of the typewriter printed text in the page 5",
            ground_truth="[the information provided in the document cannot answer this question]",
            full_final_answer=(
                'Therefore, there is no "page 5" shown, and no typewriter text on a non-existent '
                "page 5 can be identified from the given images."
            ),
        )
        is None
    )
    assert (
        _compute_rule_based_raw_final_answer_accuracy_v2(
            question="What information is given in a table?",
            ground_truth="[the information provided in the document cannot answer this question]",
            full_final_answer="There is no table present in the text. Therefore, no information is given in a table.",
        )
        is None
    )


def test_mc_judge_defers_ambiguous_long_option_discussion_to_llm():
    question = (
        "What color is the traffic light? "
        "(A) red (B) yellow (C) green (D) changing color (E) The image does not provide sufficient information."
    )
    answer = (
        "I considered (A), (B), and (C). The image has glare and blur, so the precise traffic-light color "
        "is hard to verify from the available crop."
    )

    assert (
        _raw_final_answer_matches_mc_ground_truth(
            question=question,
            ground_truth="C",
            full_final_answer=answer,
        )
        is None
    )
