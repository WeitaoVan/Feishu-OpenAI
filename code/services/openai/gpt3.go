package openai

import (
	"errors"
	"start-feishubot/logger"
	"strings"
	"fmt"

	"github.com/pkoukk/tiktoken-go"
	"github.com/pandodao/tokenizer-go"
)

type AIMode float64

const (
	Fresh      AIMode = 0.1
	Warmth     AIMode = 0.4
	Balance    AIMode = 0.7
	Creativity AIMode = 1.0
)

var AIModeMap = map[string]AIMode{
	"清新": Fresh,
	"温暖": Warmth,
	"平衡": Balance,
	"创意": Creativity,
}

var AIModeStrs = []string{
	"清新",
	"温暖",
	"平衡",
	"创意",
}

type Messages struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatGPTResponseBody 请求体
type ChatGPTResponseBody struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int                    `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatGPTChoiceItem    `json:"choices"`
	Usage   map[string]interface{} `json:"usage"`
}

type ChatGPTChoiceItem struct {
	Message      Messages `json:"message"`
	Index        int      `json:"index"`
	FinishReason string   `json:"finish_reason"`
}

// ChatGPTRequestBody 响应体
type ChatGPTRequestBody struct {
	Model            string     `json:"model"`
	Messages         []Messages `json:"messages"`
	MaxTokens        int        `json:"max_tokens"`
	Temperature      AIMode     `json:"temperature"`
	TopP             int        `json:"top_p"`
	FrequencyPenalty int        `json:"frequency_penalty"`
	PresencePenalty  int        `json:"presence_penalty"`
}

func (msg *Messages) CalculateTokenLength() int {
	text := strings.TrimSpace(msg.Content)
	return tokenizer.MustCalToken(text)
}

func (gpt *ChatGPT) Completions(msg []Messages, aiMode AIMode) (resp Messages,
	err error) {
	requestBody := ChatGPTRequestBody{
		Model:            gpt.Model,
		Messages:         msg,
		MaxTokens:        gpt.MaxTokens,
		Temperature:      aiMode,
		TopP:             1,
		FrequencyPenalty: 0,
		PresencePenalty:  0,
	}
	gptResponseBody := &ChatGPTResponseBody{}
	url := gpt.FullUrl("chat/completions")
	//fmt.Println(url)
	logger.Debug(url)
	logger.Debug("request body ", requestBody)
	if url == "" {
		return resp, errors.New("无法获取openai请求地址")
	}
	err = gpt.sendRequestWithBodyType(url, "POST", jsonBody, requestBody, gptResponseBody)
	numTokens := NumTokensFromMessagesCompletion(requestBody.Messages, requestBody.Model)
    numTokensStr := fmt.Sprintf(" (%d tokens used)", numTokens)

	if err == nil && len(gptResponseBody.Choices) > 0 {
		resp = gptResponseBody.Choices[0].Message
		resp.Content = resp.Content + numTokensStr
	} else {
		logger.Errorf("ERROR %v", err)
		resp = Messages{}
		err = errors.New("openai 请求失败")
	}
	return resp, err
}

// OpenAI Cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
func NumTokensFromMessagesCompletion(messages []Messages, model string) (numTokens int) {
	tkm, err := tiktoken.EncodingForModel(model)
	if err != nil {
		err = fmt.Errorf("encoding for model: %v", err)
		fmt.Println(err)
		return
	}

	var tokensPerMessage int
	switch model {
	case "gpt-3.5-turbo-0613",
		"gpt-3.5-turbo-16k-0613",
		"gpt-4-0314",
		"gpt-4-32k-0314",
		"gpt-4-0613",
		"gpt-4-32k-0613",
		"gpt-3.5-turbo",
		"gpt-4":
		tokensPerMessage = 3
		// tokensPerName = 1
	case "gpt-3.5-turbo-0301":
		tokensPerMessage = 4 // every message follows <|start|>{role/name}\n{content}<|end|>\n
		// tokensPerName = -1   // if there's a name, the role is omitted
	default:
		if strings.Contains(model, "gpt-3.5-turbo") {
			fmt.Println("warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
			// return NumTokensFromMessages(messages, "gpt-3.5-turbo-0613")
			return 0
		} else if strings.Contains(model, "gpt-4") {
			fmt.Println("warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
			// return NumTokensFromMessages(messages, "gpt-4-0613")
			return 0
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
		// Message here does not contain `Name`
		// numTokens += len(tkm.Encode(message.Name, nil, nil))
		// if message.Name != "" {
		// 	numTokens += tokensPerName
		// }
	}
	numTokens += 3 // every reply is primed with <|start|>assistant<|message|>
	return numTokens
}
