"""Unit tests for mcp-slim v1.5.0 semantic $ref hint preservation and disambiguation evaluation."""

from __future__ import annotations

import json
import unittest

from mcp_slim import (
    SchemaSlimmer,
    SlimStats,
    evaluate_ref_disambiguation,
    handle_server_rpc,
)


class TestV150RefSemanticHints(unittest.TestCase):
  """Verifies context-aware semantic hint retention when inlining undocumented object $refs."""

  def test_shipping_and_billing_address_disambiguation_vs_redundant_strip(self) -> None:
    tool = {
        "name": "create_checkout",
        "description": "Submit checkout with shipping and billing destinations.",
        "inputSchema": {
            "type": "object",
            "$defs": {
                "ShippingAddress": {
                    "title": "ShippingAddress",
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zip_code": {"type": "string"},
                    },
                    "required": ["street", "city", "zip_code"],
                },
                "BillingAddress": {
                    "title": "BillingAddress",
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zip_code": {"type": "string"},
                    },
                    "required": ["street", "city", "zip_code"],
                },
            },
            "properties": {
                "origin": {"$ref": "#/$defs/ShippingAddress"},
                "destination": {"$ref": "#/$defs/BillingAddress"},
                "shipping_address": {"$ref": "#/$defs/ShippingAddress"},
                "billing_address": {
                    "anyOf": [{"$ref": "#/$defs/BillingAddress"}, {"type": "null"}]
                },
            },
            "required": ["origin", "destination"],
        },
    }

    stats = SlimStats()
    slimmed = SchemaSlimmer().slim_tool(tool, stats=stats)
    props = slimmed["inputSchema"]["properties"]

    # Generic property names retain concise semantic hints from $defs names
    self.assertEqual(props["origin"].get("description"), "Shipping address")
    self.assertEqual(props["destination"].get("description"), "Billing address")

    # Self-explanatory property names strip the redundant hint at zero extra token cost
    self.assertNotIn("description", props["shipping_address"])
    self.assertNotIn("description", props["billing_address"])
    self.assertEqual(stats.ref_hints_preserved, 2)

  def test_union_branches_with_identical_fields_preserve_distinct_ref_identities(self) -> None:
    tool = {
        "name": "assign_counterparty",
        "description": "Attach either a customer or vendor record.",
        "inputSchema": {
            "type": "object",
            "$defs": {
                "Customer": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["id", "name"],
                },
                "Vendor": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["id", "name"],
                },
            },
            "properties": {
                "party": {
                    "oneOf": [
                        {"$ref": "#/$defs/Customer"},
                        {"$ref": "#/$defs/Vendor"},
                    ]
                }
            },
        },
    }

    stats = SlimStats()
    slimmed = SchemaSlimmer().slim_tool(tool, stats=stats)
    party = slimmed["inputSchema"]["properties"]["party"]
    self.assertIn("oneOf", party)
    self.assertEqual(len(party["oneOf"]), 2)
    descs = [b.get("description") for b in party["oneOf"]]
    self.assertEqual(descs, ["Customer", "Vendor"])

  def test_array_items_redundant_vs_informative_ref_hints(self) -> None:
    tool = {
        "name": "batch_recipients",
        "description": "Send invoices to recipients.",
        "inputSchema": {
            "type": "object",
            "$defs": {
                "LineItem": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                },
                "BillingContact": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                },
            },
            "properties": {
                "line_items": {"type": "array", "items": {"$ref": "#/$defs/LineItem"}},
                "targets": {"type": "array", "items": {"$ref": "#/$defs/BillingContact"}},
            },
        },
    }
    stats = SlimStats()
    slimmed = SchemaSlimmer().slim_tool(tool, stats=stats)
    props = slimmed["inputSchema"]["properties"]
    self.assertNotIn("description", props["line_items"]["items"])
    self.assertEqual(props["targets"]["items"].get("description"), "Billing contact")
    self.assertEqual(stats.ref_hints_preserved, 1)

  def test_no_ref_hints_opt_out_and_disambiguation_evaluator(self) -> None:
    report = evaluate_ref_disambiguation()
    self.assertEqual(report["semantic_hint_accuracy"], 1.0)
    self.assertLess(report["structural_only_accuracy"], 0.5)
    self.assertGreater(report["ref_hints_preserved"], 0)
    self.assertLess(report["semantic_hint_tokens"], report["raw_tokens"])

  def test_server_rpc_resource_templates_and_compact_preserve_ref_hints(self) -> None:
    tmpl_res = handle_server_rpc({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "resources/templates/list",
        "params": {},
    })
    self.assertIsNotNone(tmpl_res)
    assert tmpl_res is not None
    self.assertEqual(tmpl_res["result"], {"resourceTemplates": []})

    compact_res = handle_server_rpc({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "compact_mcp_tools",
            "arguments": {
                "tools": [{
                    "name": "ship",
                    "description": "Ship package.",
                    "inputSchema": {
                        "type": "object",
                        "$defs": {
                            "OAuth2Token": {
                                "type": "object",
                                "properties": {"access_token": {"type": "string"}},
                            }
                        },
                        "properties": {
                            "credential": {"$ref": "#/$defs/OAuth2Token"},
                        },
                    },
                }],
                "preserve_ref_hints": True,
            },
        },
    })
    assert compact_res is not None
    payload = json.loads(compact_res["result"]["content"][0]["text"])
    self.assertEqual(payload["ref_hints_preserved"], 1)
    cred_desc = payload["tools"][0]["inputSchema"]["properties"]["credential"]["description"]
    self.assertEqual(cred_desc, "OAuth2 token")


if __name__ == "__main__":
  unittest.main()
