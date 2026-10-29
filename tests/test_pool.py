from inference_proxy.pool import AUTH, NETWORK, QUOTA, SERVER, ProviderPool, load_pool_state, save_pool_state
from inference_proxy.config import ProviderConfig


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_pool(accounts: int = 2, quota_cooldown: float = 600.0) -> tuple[ProviderPool, FakeClock]:
    clock = FakeClock()
    config = ProviderConfig(
        base_url="http://zai.test/v1",
        default_model="glm-4.7",
        quota_cooldown_seconds=quota_cooldown,
        accounts=[{"api_key": f"k{i}"} for i in range(accounts)],
    )
    return ProviderPool("zai", config, clock=clock), clock


def test_acquire_rotates_accounts():
    pool, _ = make_pool()
    first = pool.acquire()
    pool.report_success(first)
    second = pool.acquire()
    pool.report_success(second)
    assert first.index != second.index


def test_acquire_prefers_least_busy():
    pool, _ = make_pool()
    first = pool.acquire()
    second = pool.acquire()
    assert {first.index, second.index} == {0, 1}
    pool.release(first)
    pool.release(second)


def test_quota_failure_cooldowns_account():
    pool, clock = make_pool(quota_cooldown=600.0)
    first = pool.acquire()
    assert first.index == 0
    pool.report_failure(first, QUOTA)

    next_account = pool.acquire()
    assert next_account.index == 1
    pool.report_success(next_account)

    assert pool.healthy_accounts()[0].index == 1
    clock.advance(599.9)
    assert pool.healthy_accounts()[0].index == 1
    clock.advance(0.2)
    assert {a.index for a in pool.healthy_accounts()} == {0, 1}


def test_all_accounts_exhausted_returns_none():
    pool, _ = make_pool(accounts=2)
    for _ in range(2):
        acc = pool.acquire()
        pool.report_failure(acc, QUOTA)
    assert pool.acquire() is None


def test_auth_failure_uses_long_cooldown():
    pool, clock = make_pool(accounts=1, quota_cooldown=60.0)
    acc = pool.acquire()
    pool.report_failure(acc, AUTH)
    assert pool.healthy_accounts() == []
    clock.advance(120)
    assert pool.healthy_accounts() == []
    clock.advance(pool.config.auth_failure_cooldown_seconds - 120 + 1)
    assert len(pool.healthy_accounts()) == 1


def test_server_failures_back_off_exponentially():
    pool, clock = make_pool(accounts=1)
    acc = pool.acquire()
    pool.report_failure(acc, SERVER)
    remaining = acc.cooldown_until - clock.now
    assert remaining == 5.0

    clock.advance(6)
    acc = pool.acquire()
    assert acc.index == 0
    pool.report_failure(acc, SERVER)
    remaining = acc.cooldown_until - clock.now
    assert remaining == 10.0


def test_success_resets_state():
    pool, clock = make_pool(quota_cooldown=60.0)
    acc = pool.acquire()
    pool.report_failure(acc, NETWORK)
    assert acc.cooldown_until > clock.now

    clock.advance(61)
    acc = pool.acquire()
    pool.report_success(acc)
    assert acc.consecutive_failures == 0
    assert acc.cooldown_until == 0.0


def test_synthesized_account_from_provider_key():
    config = ProviderConfig(base_url="http://x.test/v1", api_key="provider-key")
    pool = ProviderPool("x", config, clock=FakeClock())
    assert len(pool.accounts) == 1
    assert pool.accounts[0].api_key == "provider-key"


def test_state_round_trip(tmp_path):
    pool, clock = make_pool()
    acc = pool.acquire()
    pool.report_failure(acc, QUOTA)

    path = tmp_path / "state.json"
    save_pool_state(path, {"zai": pool})

    pool2, clock2 = make_pool()
    clock2.now = clock.now + 1
    load_pool_state(path, {"zai": pool2})
    assert pool2.accounts[0].cooldown_until == acc.cooldown_until

    clock2.advance(10_000)
    pool3, clock3 = make_pool()
    clock3.now = clock.now + 10_000
    load_pool_state(path, {"zai": pool3})
    assert pool3.accounts[0].cooldown_until == 0.0
