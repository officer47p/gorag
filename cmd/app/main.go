package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
	openai "github.com/sashabaranov/go-openai"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		panic("Missing OPENROUTER_API_KEY")
	}
	baseURL := "https://openrouter.ai/api/v1"
	// model := "deepseek/deepseek-r1-distill-llama-70b:free"

	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseURL
	client := openai.NewClientWithConfig(config)

	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
		},
	)
	if err != nil {
		fmt.Printf("ChatCompletion error: %v\n", err)
		return
	}

	fmt.Println(resp.Choices[0].Message.Content)

	req := openai.EmbeddingRequest{
		Input: []string{
			"The quick brown fox jumps over the lazy dog",
			"Embedding models are useful for search and clustering",
		},
		Model: openai.LargeEmbedding3, // or "text-embedding-3-large"
		// Optional: user ID for abuse monitoring
		// User: "user-123",
	}

	embresp, err := client.CreateEmbeddings(context.Background(), req)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Usage: %d tokens\n", embresp.Usage.TotalTokens)
	for i, embedding := range embresp.Data {
		fmt.Printf("Embedding %d (index %d): length=%d, preview=[%.6f, %.6f, ...]\n",
			i, embedding.Index, len(embedding.Embedding),
			embedding.Embedding[0], embedding.Embedding[1])
	}

	// Create an EmbeddingRequest for the user query
	queryReq := openai.EmbeddingRequest{
		Input: []string{"How many chucks would a woodchuck chuck"},
		Model: openai.AdaEmbeddingV2,
	}

	// Create an embedding for the user query
	queryResponse, err := client.CreateEmbeddings(context.Background(), queryReq)
	if err != nil {
		log.Fatal("Error creating query embedding:", err)
	}

	// Create an EmbeddingRequest for the target text
	targetReq := openai.EmbeddingRequest{
		Input: []string{"How many chucks would a woodchuck chuck if the woodchuck could chuck wood"},
		Model: openai.AdaEmbeddingV2,
	}

	// Create an embedding for the target text
	targetResponse, err := client.CreateEmbeddings(context.Background(), targetReq)
	if err != nil {
		log.Fatal("Error creating target embedding:", err)
	}

	// Now that we have the embeddings for the user query and the target text, we
	// can calculate their similarity.
	queryEmbedding := queryResponse.Data[0]
	targetEmbedding := targetResponse.Data[0]

	similarity, err := queryEmbedding.DotProduct(&targetEmbedding)
	if err != nil {
		log.Fatal("Error calculating dot product:", err)
	}

	log.Printf("The similarity score between the query and the target is %f", similarity)

	// // Your API key from OpenRouter (or Groq, Together, etc.)
	// apiKey := os.Getenv("OPENROUTER_API_KEY")
	// if apiKey == "" {
	// 	panic("Missing OPENROUTER_API_KEY")
	// }

	// // Custom client with OpenRouter base URL
	// client := openai.NewClientWithConfig(openai.ClientConfig{
	// 	BaseURL: "https://openrouter.ai/api/v1", // OpenRouter endpoint
	// 	APIKey:  apiKey,
	// 	// Optional: some providers need specific headers
	// 	HTTPClient: nil, // use default
	// 	// You can add default headers like model referral if needed
	// 	DefaultHeaders: map[string]string{
	// 		"HTTP-Referer": "https://your-app.com", // Optional, but encouraged for OpenRouter
	// 		"X-Title":      "My Awesome Go App",    // Optional
	// 	},
	// })

	// ctx := context.Background()

	// // Choose any embedding model that OpenRouter supports
	// // Full list: https://openrouter.ai/models?max_tokens=8192&supported_parameters=embeddings
	// model := openai.LargeEmbeddingModelV3 // or "text-embedding-3-large"
	// // Other popular ones:
	// // "nomic-ai/nomic-embed-text-v1.5"
	// // "snowflake-arctic-embed-l"
	// // "BAAI/bge-m3"

	// req := openai.EmbeddingRequest{
	// 	Input: []string{
	// 		"The quick brown fox jumps over the lazy dog",
	// 		"Embedding models are useful for search and clustering",
	// 	},
	// 	Model: model,
	// 	// Optional: user ID for abuse monitoring
	// 	// User: "user-123",
	// }

	// resp, err := client.CreateEmbeddings(ctx, req)
	// if err != nil {
	// 	panic(err)
	// }

	// fmt.Printf("Usage: %d tokens\n", resp.Usage.TotalTokens)
	// for i, embedding := range resp.Data {
	// 	fmt.Printf("Embedding %d (index %d): length=%d, preview=[%.6f, %.6f, ...]\n",
	// 		i, embedding.Index, len(embedding.Embedding),
	// 		embedding.Embedding[0], embedding.Embedding[1])
	// }
}
