#!/usr/bin/env bash
#
# Setup mínimo da Hyperledger Fabric test-network
# + deploy do chaincode de registo de eventos de anomalia
#
# Pré-requisitos:
#   - Docker & Docker Compose
#   - Go >= 1.21 (para compilar o chaincode)
#   - Git
#
# Uso:
#   chmod +x setup_test_network.sh
#   ./setup_test_network.sh
#
# O script assume que será executado a partir da pasta fabric/scripts
# ou que FABRIC_SAMPLES_DIR está definido.

set -euo pipefail

# =============================================================================
# Configuração
# =============================================================================

CHANNEL_NAME="${CHANNEL_NAME:-mychannel}"
CHAINCODE_NAME="${CHAINCODE_NAME:-anomalyevents}"
CHAINCODE_VERSION="${CHAINCODE_VERSION:-1.0}"
CHAINCODE_SEQUENCE="${CHAINCODE_SEQUENCE:-1}"
CHAINCODE_PATH="${CHAINCODE_PATH:-../chaincode/anomaly-events}"

# Pasta onde está (ou será colocado) o fabric-samples
FABRIC_SAMPLES_DIR="${FABRIC_SAMPLES_DIR:-$HOME/fabric-samples}"
TEST_NETWORK_DIR="${FABRIC_SAMPLES_DIR}/test-network"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_FABRIC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo " Hyperledger Fabric – Setup Test Network + Anomaly Events"
echo "============================================================"
echo "FABRIC_SAMPLES_DIR : ${FABRIC_SAMPLES_DIR}"
echo "CHANNEL_NAME       : ${CHANNEL_NAME}"
echo "CHAINCODE_NAME     : ${CHAINCODE_NAME}"
echo "CHAINCODE_PATH     : ${CHAINCODE_PATH}"
echo ""

# =============================================================================
# 1. Verificar / obter fabric-samples
# =============================================================================

if [[ ! -d "${TEST_NETWORK_DIR}" ]]; then
  echo ">>> fabric-samples não encontrado. A descarregar..."
  mkdir -p "$(dirname "${FABRIC_SAMPLES_DIR}")"
  curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
  chmod +x install-fabric.sh
  ./install-fabric.sh docker samples binary
  # O script install-fabric.sh cria fabric-samples no diretório atual
  if [[ -d "./fabric-samples" && ! -d "${FABRIC_SAMPLES_DIR}" ]]; then
    mv ./fabric-samples "${FABRIC_SAMPLES_DIR}"
  fi
  rm -f install-fabric.sh
else
  echo ">>> fabric-samples já existe em ${FABRIC_SAMPLES_DIR}"
fi

# =============================================================================
# 2. Copiar chaincode para a pasta de samples (opcional mas conveniente)
# =============================================================================

CC_DEST="${FABRIC_SAMPLES_DIR}/chaincode/anomaly-events"
echo ">>> A copiar chaincode para ${CC_DEST}"
mkdir -p "${CC_DEST}"
cp -r "${PROJECT_FABRIC_DIR}/chaincode/anomaly-events/"* "${CC_DEST}/"

# Garantir go.mod mínimo se não existir
if [[ ! -f "${CC_DEST}/go.mod" ]]; then
  cat > "${CC_DEST}/go.mod" << 'EOF'
module github.com/hyperledger/fabric-samples/anomaly-events

go 1.21

require github.com/hyperledger/fabric-contract-api-go v1.2.2
EOF
fi

# =============================================================================
# 3. Subir a test-network
# =============================================================================

cd "${TEST_NETWORK_DIR}"

echo ">>> A limpar qualquer rede anterior..."
./network.sh down || true

echo ">>> A subir a rede + criar canal ${CHANNEL_NAME}..."
./network.sh up createChannel -c "${CHANNEL_NAME}" -ca

echo ">>> Rede no ar. Contentores:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# =============================================================================
# 4. Deploy do chaincode
# =============================================================================

echo ">>> A fazer deploy do chaincode ${CHAINCODE_NAME}..."
./network.sh deployCC \
  -c "${CHANNEL_NAME}" \
  -ccn "${CHAINCODE_NAME}" \
  -ccp "${CC_DEST}" \
  -ccl go \
  -ccv "${CHAINCODE_VERSION}" \
  -ccs "${CHAINCODE_SEQUENCE}"

echo ""
echo "============================================================"
echo " Setup concluído com sucesso"
echo "============================================================"
echo ""
echo "Exemplos de invocação (a partir de ${TEST_NETWORK_DIR}):"
echo ""
echo "  # Variáveis de ambiente Org1"
echo "  export PATH=\${PWD}/../bin:\$PATH"
echo "  export FABRIC_CFG_PATH=\${PWD}/../config/"
echo "  export CORE_PEER_TLS_ENABLED=true"
echo "  export CORE_PEER_LOCALMSPID=Org1MSP"
echo "  export CORE_PEER_TLS_ROOTCERT_FILE=\${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
echo "  export CORE_PEER_MSPCONFIGPATH=\${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
echo "  export CORE_PEER_ADDRESS=localhost:7051"
echo ""
echo "  # Registar um evento"
echo "  peer chaincode invoke -o localhost:7050 \\"
echo "    --ordererTLSHostnameOverride orderer.example.com \\"
echo "    --tls --cafile \${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem \\"
echo "    -C ${CHANNEL_NAME} -n ${CHAINCODE_NAME} \\"
echo "    --peerAddresses localhost:7051 --tlsRootCertFiles \${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt \\"
echo "    --peerAddresses localhost:9051 --tlsRootCertFiles \${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt \\"
echo "    -c '{\"function\":\"RegisterEvent\",\"Args\":[\"EVT001\",\"ART123\",\"inventario\",\"stock_negativo\",\"isolation_forest\",\"0.92\",\"user_admin\",\"abc123hash\",\"Stock negativo detetado\"]}'"
echo ""
echo "  # Consultar evento"
echo "  peer chaincode query -C ${CHANNEL_NAME} -n ${CHAINCODE_NAME} \\"
echo "    -c '{\"function\":\"GetEvent\",\"Args\":[\"EVT001\"]}'"
echo ""
echo "  # Listar todos"
echo "  peer chaincode query -C ${CHANNEL_NAME} -n ${CHAINCODE_NAME} \\"
echo "    -c '{\"function\":\"GetAllEvents\",\"Args\":[]}'"
echo ""
echo "Para derrubar a rede:  cd ${TEST_NETWORK_DIR} && ./network.sh down"
echo "============================================================"
