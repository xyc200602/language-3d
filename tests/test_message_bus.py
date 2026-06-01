"""Tests for MessageBus."""

from __future__ import annotations

from lang3d.agent.message_bus import AgentMessage, MessageBus


class TestMessageBus:
    def test_publish_and_get(self):
        bus = MessageBus()
        msg = AgentMessage(sender="agent-1", type="status", payload="running")
        bus.publish(msg)
        messages = bus.get_messages()
        assert len(messages) == 1
        assert messages[0].sender == "agent-1"

    def test_filter_by_agent(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender="agent-1", type="status"))
        bus.publish(AgentMessage(sender="agent-2", type="status"))
        bus.publish(AgentMessage(sender="agent-1", type="thinking"))

        a1_msgs = bus.get_messages(agent_id="agent-1")
        assert len(a1_msgs) == 2

    def test_filter_by_type(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender="a", type="tool_call"))
        bus.publish(AgentMessage(sender="a", type="tool_result"))
        bus.publish(AgentMessage(sender="b", type="tool_call"))

        tc_msgs = bus.get_messages(type="tool_call")
        assert len(tc_msgs) == 2

    def test_filter_by_both(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender="a", type="tool_call"))
        bus.publish(AgentMessage(sender="a", type="tool_result"))
        bus.publish(AgentMessage(sender="b", type="tool_call"))

        msgs = bus.get_messages(agent_id="a", type="tool_call")
        assert len(msgs) == 1

    def test_subscribe_callback(self):
        bus = MessageBus()
        received = []
        bus.subscribe(lambda msg: received.append(msg))
        bus.publish(AgentMessage(sender="a", type="test", payload="hello"))
        assert len(received) == 1
        assert received[0].payload == "hello"

    def test_subscribe_exception_doesnt_break(self):
        bus = MessageBus()
        bus.subscribe(lambda msg: 1 / 0)  # Will raise
        good = []
        bus.subscribe(lambda msg: good.append(msg))
        bus.publish(AgentMessage(sender="a", type="test"))
        assert len(good) == 1

    def test_get_artifacts(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender="a", type="artifact", payload="/tmp/base.FCStd"))
        bus.publish(AgentMessage(sender="b", type="artifact", payload="/tmp/arm.FCStd"))
        bus.publish(AgentMessage(sender="a", type="tool_call", payload="ignored"))

        artifacts = bus.get_artifacts()
        assert len(artifacts) == 2
        assert "/tmp/base.FCStd" in artifacts
        assert "/tmp/arm.FCStd" in artifacts

    def test_get_artifacts_list_payload(self):
        bus = MessageBus()
        bus.publish(AgentMessage(
            sender="a", type="artifact",
            payload=["/tmp/part1.FCStd", "/tmp/part2.FCStd"],
        ))
        artifacts = bus.get_artifacts()
        assert len(artifacts) == 2

    def test_artifacts_deduplicated(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender="a", type="artifact", payload="/tmp/x.FCStd"))
        bus.publish(AgentMessage(sender="b", type="artifact", payload="/tmp/x.FCStd"))
        artifacts = bus.get_artifacts()
        assert len(artifacts) == 1

    def test_clear(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender="a", type="test"))
        bus.clear()
        assert len(bus.get_messages()) == 0
