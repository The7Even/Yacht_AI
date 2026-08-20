from app.ai.action_generator import Action, ActionType, DecisionResult
from app.ai.benchmark import StrategyBenchmark
from app.ai.strategy import RuleBasedStrategy
from app.core.categories import ALL_CATEGORIES


class HoldAllThenScoreStrategy:
    """Test strategy that proves benchmark rerolls actually respect held dice."""

    def __init__(self) -> None:
        self.first_dice: tuple[int, ...] | None = None
        self.second_dice: tuple[int, ...] | None = None

    def decide(self, state):
        if self.first_dice is None:
            self.first_dice = state.current_dice
            return DecisionResult(
                Action(ActionType.REROLL, (0, 1, 2, 3, 4)),
                "Hold every die once.",
            )

        if self.second_dice is None:
            self.second_dice = state.current_dice

        category = next(
            category
            for category in ALL_CATEGORIES
            if category not in state.players[state.current_player].used_categories
        )
        return DecisionResult(
            Action(ActionType.SCORE, selected_category=category),
            "Score the first available category.",
        )


def test_benchmark_is_reproducible() -> None:
    benchmark = StrategyBenchmark(seed=42)

    first = benchmark.run(RuleBasedStrategy, RuleBasedStrategy, games=4)
    second = benchmark.run(RuleBasedStrategy, RuleBasedStrategy, games=4)

    assert first == second
    assert first.games == 4
    assert first.player_one_wins + first.player_two_wins + first.draws == 4
    assert first.strategy_one_started_games == 2
    assert first.strategy_two_started_games == 2


def test_match_scores_are_valid() -> None:
    result = StrategyBenchmark(seed=7).play_match(
        RuleBasedStrategy,
        RuleBasedStrategy,
        seed=7,
    )

    assert result.strategy_one_score >= 0
    assert result.strategy_two_score >= 0
    assert result.winner in {-1, 0, 1}
    assert result.player_score >= 0
    assert result.ai_score >= 0


def test_benchmark_applies_held_indices_before_reroll() -> None:
    created: list[HoldAllThenScoreStrategy] = []

    def factory() -> HoldAllThenScoreStrategy:
        strategy = HoldAllThenScoreStrategy()
        created.append(strategy)
        return strategy

    StrategyBenchmark(seed=11).play_match(factory, factory, seed=11)

    assert created
    assert all(strategy.first_dice is not None for strategy in created)
    assert all(strategy.second_dice == strategy.first_dice for strategy in created)


def test_round_robin_returns_each_pair_once() -> None:
    results = StrategyBenchmark(seed=3).round_robin(
        (
            ("Rule A", RuleBasedStrategy),
            ("Rule B", RuleBasedStrategy),
            ("Rule C", RuleBasedStrategy),
        ),
        games_per_pair=2,
    )

    assert len(results) == 3
    assert {(result.strategy_one, result.strategy_two) for result in results} == {
        ("Rule A", "Rule B"),
        ("Rule A", "Rule C"),
        ("Rule B", "Rule C"),
    }
    assert all(result.games == 2 for result in results)
