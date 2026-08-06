"""
test_fast_engine.py
====================
Standalone sanity tests for fast_engine.py — no device/ADB needed.
Run: python test_fast_engine.py
"""

from fast_engine import FastDecider, FastVerifier, looks_atomic, parse_ui_tree

PASS = 0
FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


WHATSAPP_HOME_XML = """
<hierarchy>
  <node text="WhatsApp" content-desc="" resource-id="id/title" clickable="false" bounds="[40,80][300,140]"/>
  <node text="" content-desc="Search" resource-id="id/search" clickable="true" bounds="[900,80][1020,180]"/>
  <node text="Chats" content-desc="" resource-id="id/tab_chats" clickable="true" bounds="[0,2200][360,2340]"/>
  <node text="Mom" content-desc="" resource-id="id/chat_row" clickable="true" bounds="[40,300][1040,420]"/>
</hierarchy>
"""

CHAT_SCREEN_XML = """
<hierarchy>
  <node text="Mom" content-desc="" resource-id="id/title" clickable="false" bounds="[40,80][300,140]"/>
  <node text="" content-desc="Type a message" resource-id="id/entry" clickable="true" bounds="[40,2200][900,2340]"/>
  <node text="Send" content-desc="" resource-id="id/send_btn" clickable="true" bounds="[950,2200][1040,2340]"/>
</hierarchy>
"""


def test_open_app():
    d = FastDecider()
    action = d.decide("open whatsapp", [], "")
    check("open whatsapp -> open_app", action == {"action": "open_app", "app": "whatsapp"})


def test_tap_text_visible():
    d = FastDecider()
    action = d.decide("tap Chats", [], WHATSAPP_HOME_XML)
    check("tap Chats -> tap_text Chats", action == {"action": "tap_text", "text": "Chats"})


def test_type_instruction():
    d = FastDecider()
    action = d.decide('type "hi mom, how are you"', [], CHAT_SCREEN_XML)
    check("type instruction -> type action", action == {"action": "type", "text": "hi mom, how are you"})


def test_send_after_typing():
    d = FastDecider()
    history = [{"action": "type", "text": "hi mom"}]
    action = d.decide("send the message", history, CHAT_SCREEN_XML)
    check("send after type -> tap_text Send", action == {"action": "tap_text", "text": "Send"})


def test_visual_hint_defers():
    d = FastDecider()
    action = d.decide("tap the blue send icon", [], CHAT_SCREEN_XML)
    check("visual hint -> defers to LLM (None)", action is None)


def test_back_and_home():
    d = FastDecider()
    check("go back -> key back", d.decide("go back", [], "") == {"action": "key", "key": "back"})
    check("go home -> key home", d.decide("go home", [], "") == {"action": "key", "key": "home"})


def test_scroll():
    d = FastDecider()
    action = d.decide("scroll down to find the option", [], "")
    check("scroll down -> swipe down", action == {"action": "swipe", "direction": "down"})


def test_ambiguous_defers():
    d = FastDecider()
    action = d.decide("do the thing that makes this work", [], "")
    check("vague instruction -> defers to LLM (None)", action is None)


def test_verify_open_app():
    v = FastVerifier()
    result = v.verify(
        "open whatsapp", "Goal appears complete on screen",
        {"action": "open_app", "app": "whatsapp"},
        "com.android.launcher", "com.whatsapp",
        "", "",
    )
    check("verify open_app success", result is not None and result[0] == "complete")


def test_verify_open_app_not_yet():
    v = FastVerifier()
    result = v.verify(
        "open whatsapp", "Goal appears complete on screen",
        {"action": "open_app", "app": "whatsapp"},
        "com.android.launcher", "com.android.launcher",
        "", "",
    )
    check("verify open_app still on launcher -> defer (None)", result is None)


def test_verify_quoted_criterion():
    v = FastVerifier()
    result = v.verify(
        "search for pizza", 'Screen shows "pizza" results',
        {"action": "tap_text", "text": "Search"},
        "com.android.chrome", "com.android.chrome",
        "Chrome home", "Results for pizza nearby",
    )
    check("verify quoted criterion match -> complete", result is not None and result[0] == "complete")


def test_looks_atomic():
    check("'open whatsapp' is atomic", looks_atomic("open whatsapp") is True)
    check("'open X and then send Y' is NOT atomic", looks_atomic("open whatsapp and then send hi to mom") is False)
    check("very long goal is NOT atomic", looks_atomic(" ".join(["word"] * 20)) is False)


def test_parse_ui_tree():
    elements = parse_ui_tree(WHATSAPP_HOME_XML)
    check("parsed 4 elements", len(elements) == 4)
    clickable = [e for e in elements if e.clickable]
    check("3 clickable elements", len(clickable) == 3)
    mom = [e for e in elements if e.text == "Mom"][0]
    check("Mom center coords correct", (mom.cx, mom.cy) == (540, 360))


if __name__ == "__main__":
    test_open_app()
    test_tap_text_visible()
    test_type_instruction()
    test_send_after_typing()
    test_visual_hint_defers()
    test_back_and_home()
    test_scroll()
    test_ambiguous_defers()
    test_verify_open_app()
    test_verify_open_app_not_yet()
    test_verify_quoted_criterion()
    test_looks_atomic()
    test_parse_ui_tree()

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)
