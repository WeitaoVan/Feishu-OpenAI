package openai

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	go_openai "github.com/sashabaranov/go-openai"
	"io"
)

func (c *ChatGPT) StreamChat(ctx context.Context,
	msg []Messages, mode AIMode,
	responseStream chan string) error {
	//change msg type from Messages to openai.ChatCompletionMessage
	chatMsgs := make([]go_openai.ChatCompletionMessage, len(msg))
	for i, m := range msg {
		chatMsgs[i] = go_openai.ChatCompletionMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return c.StreamChatWithHistory(ctx, chatMsgs, 2000, mode,
		responseStream)
}

func (c *ChatGPT) StreamChatWithHistory(ctx context.Context,
	msg []go_openai.ChatCompletionMessage, maxTokens int,
	aiMode AIMode,
	responseStream chan string,
) error {

	config := go_openai.DefaultConfig(c.ApiKey[0])
	config.BaseURL = c.ApiUrl + "/v1"
	if c.Platform != OpenAI {
		baseUrl := fmt.Sprintf("https://%s.%s",
			c.AzureConfig.ResourceName, "openai.azure.com")
		config = go_openai.DefaultAzureConfig(c.AzureConfig.
			ApiToken, baseUrl)
		config.AzureModelMapperFunc = func(model string) string {
			return c.AzureConfig.DeploymentName

		}
	}

	proxyClient, parseProxyError := GetProxyClient(c.HttpProxy)
	if parseProxyError != nil {
		return parseProxyError
	}
	config.HTTPClient = proxyClient

	client := go_openai.NewClientWithConfig(config)
	//pp.Printf("client: %v", client)
	//turn aimode to float64()
	var temperature float32
	temperature = float32(aiMode)
	req := go_openai.ChatCompletionRequest{
		Model:       c.Model,
		Messages:    msg,
		N:           1,
		Temperature: temperature,
		MaxTokens:   maxTokens,
		//TopP:        1,
		//Moderation:     true,
		//ModerationStop: true,
	}
	numTokens := NumTokensFromMessages(msg, c.Model)
    numTokensStr := fmt.Sprintf(" (%d tokens)", numTokens)

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		fmt.Errorf("CreateCompletionStream returned error: %v", err)
	}

	defer stream.Close()
	for {
		response, err := stream.Recv()
		fmt.Println("response: ", response)
		if errors.Is(err, io.EOF) {
			//fmt.Println("Stream finished")
			return nil
		}
		if err != nil {
			fmt.Printf("Stream error: %v\n", err)
			return err
		}
		answer := response.Choices[0].Delta.Content + numTokensStr
		responseStream <- answer
	}
	return nil

}

// OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
func NumTokensFromMessages(messages []go_openai.ChatCompletionMessage, model string) (numTokens int) {
	tkm, err := tiktoken.EncodingForModel(model)
	if err != nil {
		err = fmt.Errorf("encoding for model: %v", err)
		fmt.Println(err)
		return
	}

	var tokensPerMessage, tokensPerName int
	switch model {
	case "gpt-3.5-turbo-0613",
		"gpt-3.5-turbo-16k-0613",
		"gpt-4-0314",
		"gpt-4-32k-0314",
		"gpt-4-0613",
		"gpt-4-32k-0613":
		tokensPerMessage = 3
		tokensPerName = 1
	case "gpt-3.5-turbo-0301":
		tokensPerMessage = 4 // every message follows <|start|>{role/name}\n{content}<|end|>\n
		tokensPerName = -1   // if there's a name, the role is omitted
	default:
		if strings.Contains(model, "gpt-3.5-turbo") {
			fmt.Println("warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
			return NumTokensFromMessages(messages, "gpt-3.5-turbo-0613")
		} else if strings.Contains(model, "gpt-4") {
			fmt.Println("warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
			return NumTokensFromMessages(messages, "gpt-4-0613")
		} else {
			err = fmt.Errorf("num_tokens_from_messages() is not implemented for model %s. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.", model)
			fmt.Println(err)
			return
		}
	}

	for _, message := range messages {
		numTokens += tokensPerMessage
		numTokens += len(tkm.Encode(message.Content, nil, nil))
		numTokens += len(tkm.Encode(message.Role, nil, nil))
		numTokens += len(tkm.Encode(message.Name, nil, nil))
		if message.Name != "" {
			numTokens += tokensPerName
		}
	}
	numTokens += 3 // every reply is primed with <|start|>assistant<|message|>
	return numTokens
}
