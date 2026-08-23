/*
 * Chaincode simples – Registo de Eventos de Anomalia Validados
 * Hyperledger Fabric (Go)
 *
 * Funções:
 *   RegisterEvent   – regista um evento validado (imutável)
 *   GetEvent        – consulta um evento por ID
 *   GetEventsByArticle – lista eventos de um código de artigo
 *   GetAllEvents    – lista todos (para demonstração / auditoria)
 *
 * Estrutura do evento:
 *   EventID, ArticleCode, Category, AnomalyType, ModelSource,
 *   Score, Timestamp, Validator, PayloadHash, Status
 */

package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// AnomalyEvent representa um evento de anomalia validado
type AnomalyEvent struct {
	EventID      string  `json:"eventId"`
	ArticleCode  string  `json:"articleCode"`
	Category     string  `json:"category"`     // configuracao_receita | padrao_consumo | inventario
	AnomalyType  string  `json:"anomalyType"`  // tipo específico (real ou sintético)
	ModelSource  string  `json:"modelSource"`  // isolation_forest | autoencoder | hybrid | manual
	Score        float64 `json:"score"`
	Timestamp    string  `json:"timestamp"`    // RFC3339
	Validator    string  `json:"validator"`    // utilizador / sistema que validou
	PayloadHash  string  `json:"payloadHash"`  // hash do registo original (integridade)
	Status       string  `json:"status"`       // VALIDATED | REJECTED | PENDING
	Description  string  `json:"description"`
}

// SmartContract fornece as funções do chaincode
type SmartContract struct {
	contractapi.Contract
}

// RegisterEvent regista um novo evento de anomalia validado
func (s *SmartContract) RegisterEvent(
	ctx contractapi.TransactionContextInterface,
	eventID string,
	articleCode string,
	category string,
	anomalyType string,
	modelSource string,
	score float64,
	validator string,
	payloadHash string,
	description string,
) error {

	exists, err := s.EventExists(ctx, eventID)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("evento %s já existe", eventID)
	}

	event := AnomalyEvent{
		EventID:     eventID,
		ArticleCode: articleCode,
		Category:    category,
		AnomalyType: anomalyType,
		ModelSource: modelSource,
		Score:       score,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		Validator:   validator,
		PayloadHash: payloadHash,
		Status:      "VALIDATED",
		Description: description,
	}

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(eventID, eventJSON)
}

// GetEvent devolve um evento pelo ID
func (s *SmartContract) GetEvent(ctx contractapi.TransactionContextInterface, eventID string) (*AnomalyEvent, error) {
	eventJSON, err := ctx.GetStub().GetState(eventID)
	if err != nil {
		return nil, fmt.Errorf("falha ao ler estado: %v", err)
	}
	if eventJSON == nil {
		return nil, fmt.Errorf("evento %s não existe", eventID)
	}

	var event AnomalyEvent
	err = json.Unmarshal(eventJSON, &event)
	if err != nil {
		return nil, err
	}
	return &event, nil
}

// EventExists verifica se um evento já está registado
func (s *SmartContract) EventExists(ctx contractapi.TransactionContextInterface, eventID string) (bool, error) {
	eventJSON, err := ctx.GetStub().GetState(eventID)
	if err != nil {
		return false, fmt.Errorf("falha ao ler estado: %v", err)
	}
	return eventJSON != nil, nil
}

// GetEventsByArticle devolve todos os eventos de um código de artigo
func (s *SmartContract) GetEventsByArticle(ctx contractapi.TransactionContextInterface, articleCode string) ([]*AnomalyEvent, error) {
	query := fmt.Sprintf(`{"selector":{"articleCode":"%s"}}`, articleCode)
	return s.queryEvents(ctx, query)
}

// GetAllEvents devolve todos os eventos (uso apenas em redes de teste)
func (s *SmartContract) GetAllEvents(ctx contractapi.TransactionContextInterface) ([]*AnomalyEvent, error) {
	// Range query sobre todas as keys
	iterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer iterator.Close()

	var events []*AnomalyEvent
	for iterator.HasNext() {
		result, err := iterator.Next()
		if err != nil {
			return nil, err
		}
		var event AnomalyEvent
		err = json.Unmarshal(result.Value, &event)
		if err != nil {
			return nil, err
		}
		events = append(events, &event)
	}
	return events, nil
}

func (s *SmartContract) queryEvents(ctx contractapi.TransactionContextInterface, query string) ([]*AnomalyEvent, error) {
	iterator, err := ctx.GetStub().GetQueryResult(query)
	if err != nil {
		return nil, err
	}
	defer iterator.Close()

	var events []*AnomalyEvent
	for iterator.HasNext() {
		result, err := iterator.Next()
		if err != nil {
			return nil, err
		}
		var event AnomalyEvent
		err = json.Unmarshal(result.Value, &event)
		if err != nil {
			return nil, err
		}
		events = append(events, &event)
	}
	return events, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Erro ao criar chaincode: %v\n", err)
		return
	}
	if err := chaincode.Start(); err != nil {
		fmt.Printf("Erro ao iniciar chaincode: %v\n", err)
	}
}
