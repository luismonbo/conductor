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


def test_thresholds_calibrated_on_decision_1():
    assert (EXTRACTION.threshold, PRIVATE_INFO.threshold, OFF_TOPIC.threshold) == (
        0.70, 0.70, 0.90,
    )


def test_extraction_and_private_info_share_the_not_shareable_refusal():
    assert EXTRACTION.refusal == PRIVATE_INFO.refusal == "not_shareable"
    assert OFF_TOPIC.refusal == "off_topic"


def test_every_check_targets_the_visitor_message_and_has_examples_on_both_sides():
    for check in BATTERY:
        assert "`visitor_message`" in check.question
        assert check.yes.examples and check.no.examples


def test_version():
    assert BATTERY_VERSION == "v1"
