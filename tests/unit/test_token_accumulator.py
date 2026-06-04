from harness.observability.token_accumulator import TokenAccumulator


def test_zero_defaults():
    acc = TokenAccumulator()
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.iterations == 0


def test_accumulates_across_multiple_calls():
    acc = TokenAccumulator()
    acc.add(input=100, output=50)
    acc.add(input=200, output=75)
    assert acc.input_tokens == 300
    assert acc.output_tokens == 125
    assert acc.iterations == 2


def test_add_with_zero_values():
    acc = TokenAccumulator()
    acc.add(input=0, output=0)
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.iterations == 1
