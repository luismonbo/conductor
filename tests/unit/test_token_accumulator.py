from harness.observability.token_accumulator import TokenAccumulator


def test_zero_defaults():
    acc = TokenAccumulator()
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.iterations == 0


def test_single_call_with_values():
    acc = TokenAccumulator()
    acc.add(input_tokens=100, output_tokens=50)
    assert acc.input_tokens == 100
    assert acc.output_tokens == 50
    assert acc.total_tokens == 150
    assert acc.iterations == 1


def test_accumulates_across_multiple_calls():
    acc = TokenAccumulator()
    acc.add(input_tokens=100, output_tokens=50)
    acc.add(input_tokens=200, output_tokens=75)
    assert acc.input_tokens == 300
    assert acc.output_tokens == 125
    assert acc.total_tokens == 425
    assert acc.iterations == 2


def test_add_with_zero_values():
    acc = TokenAccumulator()
    acc.add(input_tokens=0, output_tokens=0)
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.total_tokens == 0
    assert acc.iterations == 1
