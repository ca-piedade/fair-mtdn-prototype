#!/usr/bin/env python3
"""
RQ2 -- prepares the Hyperledger Fabric event registration commands for the
augmented real-data evaluation set (Section 4.5 / Table 4.3).

Adapted from 11-Tentativa-Validacao-Dados-Reais-Excluida-Ch5/fabric/scripts/
register_validated_events.py (old pipeline, different column schema) to read
flagged_events_augmented.csv (produced by export_rq2_events_augmented.py),
which holds the SAME Isolation Forest best-F1 held-out predictions reported
in Table 4.3.

Does NOT talk to Fabric itself -- generates:
  1. events_to_register.json  -- the event payloads
  2. invoke_register_events.sh -- ready-to-paste `peer chaincode invoke`
     commands, to run from fabric-samples/test-network after
     setup_test_network.sh has deployed the `anomalyevents` chaincode.

Usage
-----
    python3 register_events_augmented.py --max-events 15

Run export_rq2_events_augmented.py FIRST if flagged_events_augmented.csv is
missing or stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fabric-register-augmented")


def payload_hash(row: pd.Series) -> str:
    raw = row.to_json(date_format="iso")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_event(row: pd.Series, idx: int) -> Dict[str, Any]:
    event_id = f"EVT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{idx:05d}"

    category = row.get("category", "none")
    if pd.isna(category) or category == "none":
        category = "unclassified"  # flagged by the model but not a true injected anomaly (FP)

    article = str(row.get("article_code", f"UNKNOWN-{idx}"))
    score = float(row.get("if_score", 0.0))

    return {
        "eventId": event_id,
        "articleCode": article,
        "category": str(category),
        "anomalyType": "single_feature_zscore_deviation",
        "modelSource": "isolation_forest",
        "score": round(score, 6),
        "validator": "system_auto",
        "payloadHash": payload_hash(row),
        "description": (
            f"Isolation Forest flag (best-F1 threshold, held-out test partition, "
            f"Table 4.3) -- article={article}, deviation_zscore_historical="
            f"{row.get('deviation_zscore_historical', float('nan')):.3f}"
        ),
    }


def generate_peer_commands(
    events: List[Dict[str, Any]],
    channel: str = "mychannel",
    chaincode: str = "anomalyevents",
) -> List[str]:
    cmds = []
    for ev in events:
        args = [
            ev["eventId"], ev["articleCode"], ev["category"], ev["anomalyType"],
            ev["modelSource"], str(ev["score"]), ev["validator"], ev["payloadHash"],
            ev["description"],
        ]
        c_payload = {"function": "RegisterEvent", "Args": args}
        c_str = json.dumps(c_payload, ensure_ascii=False)

        cmd = (
            f'peer chaincode invoke -o localhost:7050 '
            f'--ordererTLSHostnameOverride orderer.example.com '
            f'--tls --cafile ${{PWD}}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem '
            f'-C {channel} -n {chaincode} '
            f'--peerAddresses localhost:7051 --tlsRootCertFiles ${{PWD}}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt '
            f'--peerAddresses localhost:9051 --tlsRootCertFiles ${{PWD}}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt '
            f"-c '{c_str}'"
        )
        cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preparar registo de eventos augmented na Fabric")
    parser.add_argument("--events-csv", type=Path, default=Path("flagged_events_augmented.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--max-events", type=int, default=15,
                         help="Número de eventos a preparar (mantém pequeno para a demo/screenshots)")
    parser.add_argument("--channel", default="mychannel")
    parser.add_argument("--chaincode", default="anomalyevents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.events_csv.exists():
        raise FileNotFoundError(
            f"{args.events_csv} não encontrado -- corre export_rq2_events_augmented.py primeiro."
        )

    df = pd.read_csv(args.events_csv)
    df = df.head(args.max_events)  # already sorted by if_score desc
    logger.info(f"A preparar {len(df)} eventos (de {args.events_csv}, top by IF score).")

    events = [build_event(row, i) for i, (_, row) in enumerate(df.iterrows())]

    events_path = args.output_dir / "events_to_register.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    logger.info(f"Eventos JSON: {events_path}")

    cmds = generate_peer_commands(events, channel=args.channel, chaincode=args.chaincode)
    script_path = args.output_dir / "invoke_register_events.sh"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Executar a partir de fabric-samples/test-network, com as env vars Org1 ativas\n")
        f.write("set -euo pipefail\n\n")
        for cmd in cmds:
            f.write(cmd + "\n")
            f.write("sleep 1\n")
    script_path.chmod(0o755)
    logger.info(f"Script de invocação: {script_path}")

    logger.info("Concluído. Segue o guião passo a passo para: 1) setup_test_network.sh, "
                "2) deploy do chaincode, 3) correr invoke_register_events.sh.")


if __name__ == "__main__":
    main()
