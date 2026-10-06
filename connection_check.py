"""
Video connection check -- for when the live video will not connect on Streamlit Cloud.

The live video travels between the browser and this server by WebRTC. To find a path, each side collects
addresses where it can be reached:
  * through a STUN server it learns its public address -- this only works if UDP traffic is allowed;
  * through a TURN server it gets a "relay" address -- the TURN server forwards the video, which works
    through most firewalls (TURN over TCP even where UDP is blocked).
This check runs ON THE SERVER and tries every STUN and TURN address in the app's settings, so you can see
which ones work from the cloud. (The browser's side is not tested here: in Chrome, open
chrome://webrtc-internals in another tab while the video connects.)
"""
import asyncio

import aioice
from aioice.turn import create_turn_endpoint
from aiortc.rtcicetransport import parse_stun_turn_uri

TIMEOUT_SECONDS = 10


async def test_stun(host, port):
    """Ask a STUN server for this server's public address. Returns a message, or raises an error."""
    connection = aioice.Connection(ice_controlling=True, stun_server=(host, port))
    try:
        await asyncio.wait_for(connection.gather_candidates(), timeout=TIMEOUT_SECONDS)
        for candidate in connection.local_candidates:
            if candidate.type == "srflx":                  # srflx = the public address the STUN server saw
                return "public address " + candidate.host
        raise RuntimeError("no answer from the STUN server (UDP traffic may be blocked)")
    finally:
        await connection.close()


async def test_turn(host, port, username, password, transport, use_tls):
    """Log in to a TURN server and ask for a relay address. Returns a message, or raises an error."""
    relay, _ = await asyncio.wait_for(
        create_turn_endpoint(asyncio.DatagramProtocol, server_addr=(host, port), username=username,
                             password=password, ssl=use_tls, transport=transport),
        timeout=TIMEOUT_SECONDS)
    address = relay.get_extra_info("sockname")
    relay.close()                                          # give the relay address back to the TURN server
    await asyncio.sleep(1)                                 # ... and wait for it to say goodbye
    return "relay address " + str(address[0]) + ":" + str(address[1])


def check_video_connection(ice_servers):
    """Try every STUN and TURN address in ice_servers, from this server. Returns a list of (ok, message)."""
    results = []
    for server in ice_servers:
        urls = server["urls"]
        if isinstance(urls, str):                          # one address, or a list of addresses
            urls = [urls]
        for url in urls:
            parsed = parse_stun_turn_uri(url)
            if parsed["scheme"] == "stun":
                test = test_stun(parsed["host"], parsed["port"])
            else:
                test = test_turn(parsed["host"], parsed["port"], server.get("username"),
                                 server.get("credential"), parsed["transport"], parsed["scheme"] == "turns")
            try:
                message = asyncio.run(test)
                results.append((True, url + "  ->  works: " + message))
            except Exception as error:
                reason = str(error)
                if reason == "":
                    reason = type(error).__name__          # e.g. TimeoutError: no answer at all
                results.append((False, url + "  ->  FAILED: " + reason))
    return results
