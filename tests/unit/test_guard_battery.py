"""Pin the v1 battery: order, thresholds, and refusals that never reveal which rule fired."""
from harness.security.battery import (
    BATTERY,
    BATTERY_VERSION,
    EXTRACTION,
    OFF_TOPIC,
    PRIVATE_INFO,
)


def test_battery_order_is_the_precedence_order():
    assert [check.id for check in BATTERY] == ["extraction", "private_info", "off_topic"]


def test_starting_thresholds():
    assert (EXTRACTION.threshold, PRIVATE_INFO.threshold, OFF_TOPIC.threshold) == (
        0.70, 0.70, 0.80,
    )


def test_extraction_and_private_info_share_one_refusal():
    assert EXTRACTION.refusal == PRIVATE_INFO.refusal != OFF_TOPIC.refusal


def test_the_shared_refusal_sends_salary_and_rate_questions_to_the_contact_links():
    assert "salary or rates" in PRIVATE_INFO.refusal
    assert "contact links on this site" in PRIVATE_INFO.refusal


def test_every_check_targets_the_visitor_message_and_has_examples_on_both_sides():
    for check in BATTERY:
        assert "`visitor_message`" in check.question
        assert check.yes.examples and check.no.examples


def test_version():
    assert BATTERY_VERSION == "v1"
