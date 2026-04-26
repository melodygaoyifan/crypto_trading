"""
shared_state_registry.py — generic "module-level singleton state shared
across instances by key" helper.

The pattern this codifies (P68 — cross-instance nonce ratchet drift):

    Multiple object instances need to coordinate on a piece of state
    that conceptually belongs to a SHARED RESOURCE (an external API
    that tracks a per-key counter, a file on disk, a remote service).
    Each instance has its own in-memory copy of the state. The state
    is supposed to stay in sync across instances, but in practice it
    drifts because each instance only persists periodically.

    The CORRECT pattern: hold the state in a module-level dict keyed
    by a stable identifier (API key fingerprint, file path, etc.).
    All instances with the same key look up the SAME state holder.

The pattern has appeared (or should appear) in this codebase:
  - infra/kraken_rest_client.py P68 — cross-instance nonce ratchet.
    Currently uses an inline `_SHARED_NONCE_STATES` dict + ad-hoc
    `_get_or_init_shared_nonce` classmethod. Could migrate to this
    helper for consistency.
  - Latent: any singleton-like state (cascade_governor, failure_memory,
    confidence_scorer) where two callers reach for the same global —
    this helper makes the intent explicit.

Why "registry" not "singleton"
-------------------------------
Classical singleton = one instance per process. This pattern is
"one shared-state object per (registry, key)" — a parameterized
singleton. Two clients with API key A share state, but a third
client with API key B has its OWN state. The registry is what makes
the parameterization explicit.

Usage
-----

    from infra.shared_state_registry import SharedRegistry

    # Define a holder type for your shared state.
    @dataclass
    class _NonceState:
        ratchet_ms: int = 0
        dirty: bool = False

    # Module-level registry — one per "kind of shared state".
    _NONCE_REGISTRY = SharedRegistry[_NonceState](
        factory=lambda: _NonceState(),
        label="kraken_nonce",
    )

    # Inside your client class:
    class KrakenClient:
        def __init__(self, api_key: str, state_path: Path):
            fingerprint = sha256(api_key.encode()).hexdigest()[:16]
            # All clients with the same (fingerprint, state_path) get the
            # same _NonceState object.
            self._shared_nonce = _NONCE_REGISTRY.get_or_create(
                key=(fingerprint, str(state_path.resolve())),
                # Optional: per-key initialization callback. Runs ONCE on
                # first lookup of this key. Subsequent lookups skip it.
                first_init=lambda holder: self._load_persisted_into(holder),
            )

        def use_state(self):
            with self._shared_nonce.lock:
                self._shared_nonce.value.ratchet_ms += 1

Each holder also carries a per-key threading.RLock so callers can
coordinate access to the value without managing the lock separately.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Hashable, Optional, TypeVar

V = TypeVar("V")
K = TypeVar("K", bound=Hashable)


@dataclass
class SharedHolder(Generic[V]):
    """One shared-state slot. The `value` is the user's actual state
    object (any type). The `lock` is a per-key RLock the user can take
    to coordinate read/write access."""
    value: V
    lock: threading.RLock = field(default_factory=threading.RLock)
    # Set on first creation; lets first_init callbacks know they're the
    # one-time initializer (vs later get_or_create calls that find the
    # existing holder).
    first_created: bool = True


class SharedRegistry(Generic[V]):
    """Module-level dict of (key → SharedHolder). All instances with
    the same key get the same holder.

    Construct one of these per *kind* of shared state, at module level.
    Don't instantiate per-call — the whole point is that the registry
    persists across the process lifetime.

    Type parameter V is the held value type.
    """

    def __init__(
        self,
        factory: Callable[[], V],
        label: str = "registry",
    ) -> None:
        """
        Args:
            factory: zero-arg callable that returns a fresh value of type
                V. Called ONCE per key on first get_or_create.
            label: short string for use in debug output / logging.
        """
        self._factory = factory
        self._label = label
        self._holders: Dict[Hashable, SharedHolder[V]] = {}
        self._registry_lock = threading.Lock()

    def get_or_create(
        self,
        key: Hashable,
        first_init: Optional[Callable[[SharedHolder[V]], None]] = None,
    ) -> SharedHolder[V]:
        """Return the SharedHolder for this key. Creates it if absent.

        Args:
            key: anything Hashable. Tuples are common (e.g. (fingerprint,
                path_str)). Use the SAME key from every caller that
                should share state.
            first_init: optional callback invoked ONCE, on the first call
                for this key, AFTER the holder is created but BEFORE the
                lock is released. Used to warm up the holder from disk /
                remote state. Subsequent calls find the existing holder
                and skip first_init. The callback receives the holder so
                it can mutate `holder.value` in place.

        Returns:
            SharedHolder. All future get_or_create(key=...) calls return
            the same instance.
        """
        with self._registry_lock:
            existing = self._holders.get(key)
            if existing is not None:
                # Mark as not-first-created so callers can detect
                # whether they raced to create vs found existing.
                existing.first_created = False
                return existing
            holder = SharedHolder(value=self._factory(), first_created=True)
            self._holders[key] = holder
            if first_init is not None:
                # Callback runs INSIDE the registry lock so concurrent
                # creators see the fully-initialized holder. Keep
                # callbacks fast (read disk, populate dict — not slow
                # network calls).
                try:
                    first_init(holder)
                except Exception:
                    # If first_init crashes, evict the partial holder
                    # so the next caller can retry from scratch.
                    self._holders.pop(key, None)
                    raise
            return holder

    def has(self, key: Hashable) -> bool:
        with self._registry_lock:
            return key in self._holders

    def evict(self, key: Hashable) -> bool:
        """Remove the holder for `key`. Returns True if it existed.

        Use cases: testing, key rotation. NOT for runtime cleanup —
        evicting while another caller holds the lock will not affect
        their reference (they still have the old holder); subsequent
        lookups create a new one. So evict is "rotate the key", not
        "synchronize a shared deletion".
        """
        with self._registry_lock:
            return self._holders.pop(key, None) is not None

    def size(self) -> int:
        """Number of distinct keys currently in the registry."""
        with self._registry_lock:
            return len(self._holders)

    def keys(self) -> list:
        """Snapshot of current keys. Don't iterate while modifying the
        registry from another thread."""
        with self._registry_lock:
            return list(self._holders.keys())

    def reset(self) -> None:
        """Drop all holders. For test isolation. NEVER call this from
        production code — outstanding references to existing holders
        will be orphaned (still valid, just no longer in the registry)."""
        with self._registry_lock:
            self._holders.clear()

    def __repr__(self) -> str:
        return f"<SharedRegistry label={self._label!r} size={self.size()}>"
