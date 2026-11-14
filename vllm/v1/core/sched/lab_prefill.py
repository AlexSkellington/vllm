"""Best-effort, request-scoped vLLM scheduler telemetry. Never affects inference."""
import json
import re
import socket
import time

SOCKET_PATH = '/run/vllm-prefill-progress/progress.sock'
REQUEST_ID = re.compile(r'^(chatcmpl|cmpl)-hlp-([0-9a-f]{32})(-0)?(?:-[0-9a-f]{8})?$')
try:
    _socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    _socket.setblocking(False)
except (OSError, AttributeError):
    _socket = None


def completed(request, output_is_stale):
    """Publish completed prompt position, never optimistic scheduled work."""
    try:
        if _socket is None or output_is_stale or request.num_preemptions:
            return
        match = REQUEST_ID.fullmatch(request.request_id)
        stats = request.prefill_stats
        if (not match or stats is None or
                (match.group(1) == 'cmpl') != bool(match.group(3))):
            return
        total = request.num_prompt_tokens
        cache = stats.num_cached_tokens
        if stats.num_prompt_tokens != total or total <= 0:
            return
        position = min(total, request.num_computed_tokens - request.num_in_flight_tokens)
        if position <= cache or cache < 0:
            return
        now = time.monotonic()
        if now - getattr(request, '_lab_prefill_last_sent', 0.0) < .25:
            return
        request._lab_prefill_last_sent = now
        payload = {'id': match.group(2), 'position': position, 'total': total,
                   'cache': cache, 'at': now, 'preemptions': request.num_preemptions}
        _socket.sendto(json.dumps(payload, separators=(',', ':')).encode(), SOCKET_PATH)
    except Exception:
        # Telemetry is optional, bounded and never allowed to interrupt a request.
        return
