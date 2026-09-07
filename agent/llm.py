import os

try:
    # The installed openai SDK's `http_client` param expects this specific httpx variant
    # (it moved to a new major "httpx2" package) - importing whichever one is actually
    # present keeps this working whether or not that migration has happened in a given
    # environment's pinned dependency versions.
    import httpx2 as _uds_httpx
except ImportError:
    import httpx as _uds_httpx
import openai
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.config import (
    HIGH_MODEL,
    LOCAL_MODEL,
    LOW_MODEL,
    MID_MODEL,
    OMNIROUTE_API_KEY,
    OMNIROUTE_BASE_URL,
)

# Set only inside self-improvement authoring/eval containers (--network none): routes every
# model call through the trusted host broker over a Unix domain socket instead of a normal TCP
# connection to OmniRoute. The broker holds the real OMNIROUTE_API_KEY; this process never sees
# it. The broker's route surface (POST /v1/chat/completions, GET /v1/models) is unchanged by
# this - only the transport differs, so the openai SDK's URL construction stays the same.
BROKER_SOCKET_PATH = os.getenv("BROKER_SOCKET_PATH")

TIER_MODELS = {"low": LOW_MODEL, "mid": MID_MODEL, "high": HIGH_MODEL, "local": LOCAL_MODEL}
NEXT_TIER = {"low": "mid", "mid": "high", "high": "local"}

# OmniRoute pools multiple providers behind one combo (e.g. auto/cheap) and
# tries them in sequence. Individual providers in the pool can be transiently
# broken (missing auth, rate limited, etc); the gateway itself recommends
# retrying the same combo in that case, so we retry a few times before giving up.
# RateLimitError gets its own, much more patient retry below: a "cooldown" (observed
# lasting anywhere from a few seconds to a couple of minutes) is usually worth waiting
# out on the SAME tier rather than immediately escalating to a possibly-equally-exhausted
# one, whereas other infra errors are treated as quick, likely-permanent-for-now failures.
FAST_INFRA_ERRORS = (
    openai.BadRequestError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
INFRA_ERRORS = FAST_INFRA_ERRORS + (openai.RateLimitError,)


def get_client(model: str, temperature: float = 0) -> ChatOpenAI:
    if BROKER_SOCKET_PATH:
        return ChatOpenAI(
            base_url="http://broker/v1",
            api_key="unused-broker-does-not-check-this",
            model=model,
            temperature=temperature,
            http_client=_uds_httpx.Client(
                transport=_uds_httpx.HTTPTransport(uds=BROKER_SOCKET_PATH)
            ),
        )
    return ChatOpenAI(
        base_url=OMNIROUTE_BASE_URL,
        api_key=OMNIROUTE_API_KEY,
        model=model,
        temperature=temperature,
    )


def client_for_tier(tier: str, temperature: float = 0) -> ChatOpenAI:
    return get_client(TIER_MODELS[tier], temperature=temperature)


@retry(
    retry=retry_if_exception_type(FAST_INFRA_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _invoke_fast_retry(runnable, input):
    return runnable.invoke(input)


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    reraise=True,
)
def invoke_with_retry(runnable, input):
    return _invoke_fast_retry(runnable, input)
