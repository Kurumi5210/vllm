// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	sequentialMode = "sequential"
	concurrentMode = "concurrent"
)

var (
	completionEndpoints = map[string]struct{}{
		"/v1/completions":      {},
		"/v1/chat/completions": {},
	}
	hopByHopHeaders = map[string]struct{}{
		"connection":          {},
		"keep-alive":          {},
		"proxy-authenticate":  {},
		"proxy-authorization": {},
		"te":                  {},
		"trailer":             {},
		"transfer-encoding":   {},
		"upgrade":             {},
	}
	streamBufferPool = sync.Pool{
		New: func() any {
			buffer := make([]byte, 32*1024)
			return &buffer
		},
	}
)

type prefillSpec struct {
	rawURL        string
	bootstrapPort int
}

type config struct {
	host              string
	port              int
	prefill           []prefillSpec
	decode            []string
	prefillDecodeMode string
}

type prefillClient struct {
	rawURL       string
	baseURL      *url.URL
	bootstrapURL string
	dpEngineIDs  map[int]any
}

type prefillTarget struct {
	client *prefillClient
	dpRank int
}

type accessLogResponseWriter struct {
	http.ResponseWriter
	statusCode   int
	bytesWritten int64
}

func (writer *accessLogResponseWriter) WriteHeader(statusCode int) {
	if writer.statusCode != 0 {
		return
	}
	writer.statusCode = statusCode
	writer.ResponseWriter.WriteHeader(statusCode)
}

func (writer *accessLogResponseWriter) Write(data []byte) (int, error) {
	if writer.statusCode == 0 {
		writer.WriteHeader(http.StatusOK)
	}
	written, err := writer.ResponseWriter.Write(data)
	writer.bytesWritten += int64(written)
	return written, err
}

func (writer *accessLogResponseWriter) Flush() {
	if writer.statusCode == 0 {
		writer.WriteHeader(http.StatusOK)
	}
	if flusher, ok := writer.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (writer *accessLogResponseWriter) Unwrap() http.ResponseWriter {
	return writer.ResponseWriter
}

type proxy struct {
	client            *http.Client
	prefillClients    []*prefillClient
	prefillTargets    []prefillTarget
	decodeURLs        []*url.URL
	prefillDecodeMode string
	ready             atomic.Bool
	prefillIndex      atomic.Uint64
	decodeIndex       atomic.Uint64
}

func main() {
	cfg, err := parseArgs(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		printUsage(os.Stdout)
		return
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		printUsage(os.Stderr)
		os.Exit(2)
	}

	p, err := newProxy(cfg)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf(
		"Got %d prefill clients and %d decode clients.",
		len(p.prefillClients), len(p.decodeURLs),
	)
	go p.initializePrefillers()

	address := net.JoinHostPort(cfg.host, strconv.Itoa(cfg.port))
	server := &http.Server{
		Addr:    address,
		Handler: p,
	}
	log.Printf(
		"Mooncake connector proxy listening on %s (mode=%s)",
		address, cfg.prefillDecodeMode,
	)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func parseArgs(args []string) (config, error) {
	cfg := config{
		host:              "127.0.0.1",
		port:              8000,
		prefillDecodeMode: sequentialMode,
	}

	for i := 0; i < len(args); i++ {
		name, value, hasValue := strings.Cut(args[i], "=")
		nextValue := func() (string, error) {
			if hasValue {
				return value, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", name)
			}
			i++
			return args[i], nil
		}

		switch name {
		case "-h", "--help":
			return config{}, flag.ErrHelp
		case "--host":
			parsed, err := nextValue()
			if err != nil {
				return config{}, err
			}
			cfg.host = parsed
		case "--port":
			parsed, err := nextValue()
			if err != nil {
				return config{}, err
			}
			cfg.port, err = strconv.Atoi(parsed)
			if err != nil {
				return config{}, fmt.Errorf("invalid --port %q: %w", parsed, err)
			}
		case "--decode":
			parsed, err := nextValue()
			if err != nil {
				return config{}, err
			}
			cfg.decode = append(cfg.decode, parsed)
		case "--prefill":
			parsed, err := nextValue()
			if err != nil {
				return config{}, err
			}
			spec := prefillSpec{rawURL: parsed}
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				i++
				bootstrapPort := args[i]
				if !strings.EqualFold(bootstrapPort, "none") {
					spec.bootstrapPort, err = strconv.Atoi(bootstrapPort)
					if err != nil {
						return config{}, fmt.Errorf(
							"invalid bootstrap port %q: must be a number or 'none'",
							bootstrapPort,
						)
					}
				}
			}
			cfg.prefill = append(cfg.prefill, spec)
		case "--mode":
			parsed, err := nextValue()
			if err != nil {
				return config{}, err
			}
			cfg.prefillDecodeMode = parsed
		default:
			return config{}, fmt.Errorf("unknown argument %q", name)
		}
	}

	if cfg.prefillDecodeMode != sequentialMode &&
		cfg.prefillDecodeMode != concurrentMode {
		return config{}, fmt.Errorf(
			"invalid --mode %q: want %q or %q",
			cfg.prefillDecodeMode, sequentialMode, concurrentMode,
		)
	}
	if len(cfg.prefill) == 0 {
		return config{}, errors.New("at least one --prefill is required")
	}
	if len(cfg.decode) == 0 {
		return config{}, errors.New("at least one --decode is required")
	}
	return cfg, nil
}

func printUsage(output io.Writer) {
	fmt.Fprintln(output, "Usage: mooncake_connector_proxy [options]")
	fmt.Fprintln(output, "")
	fmt.Fprintln(output, "Options:")
	fmt.Fprintln(output, "  --host HOST                         listen host (default 127.0.0.1)")
	fmt.Fprintln(output, "  --port PORT                         listen port (default 8000)")
	fmt.Fprintln(output, "  --prefill URL [BOOTSTRAP_PORT]      prefill URL; may be repeated")
	fmt.Fprintln(output, "  --decode URL                        decode URL; may be repeated")
	fmt.Fprintln(output, "  --mode MODE                         sequential or concurrent (default sequential)")
	fmt.Fprintln(output, "  -h, --help                          show this help")
}

func newProxy(cfg config) (*proxy, error) {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 4096
	transport.MaxIdleConnsPerHost = 1024
	transport.MaxConnsPerHost = 0
	transport.DisableCompression = true
	client := &http.Client{
		Transport: transport,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	p := &proxy{
		client:            client,
		prefillDecodeMode: cfg.prefillDecodeMode,
	}
	for _, spec := range cfg.prefill {
		baseURL, err := parseServiceURL(spec.rawURL)
		if err != nil {
			return nil, fmt.Errorf("invalid prefill URL %q: %w", spec.rawURL, err)
		}
		bootstrapPort := spec.bootstrapPort
		if bootstrapPort == 0 {
			bootstrapPort = 8998
		}
		bootstrapURL := "http://" + net.JoinHostPort(
			baseURL.Hostname(), strconv.Itoa(bootstrapPort),
		)
		p.prefillClients = append(p.prefillClients, &prefillClient{
			rawURL:       spec.rawURL,
			baseURL:      baseURL,
			bootstrapURL: bootstrapURL,
			dpEngineIDs:  make(map[int]any),
		})
	}
	for _, rawURL := range cfg.decode {
		baseURL, err := parseServiceURL(rawURL)
		if err != nil {
			return nil, fmt.Errorf("invalid decode URL %q: %w", rawURL, err)
		}
		p.decodeURLs = append(p.decodeURLs, baseURL)
	}
	return p, nil
}

func parseServiceURL(rawURL string) (*url.URL, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, errors.New("URL scheme must be http or https")
	}
	if parsed.Hostname() == "" {
		return nil, errors.New("URL host is empty")
	}
	return parsed, nil
}

func (p *proxy) initializePrefillers() {
	for _, prefill := range p.prefillClients {
		for {
			response, err := p.client.Get(serviceURL(prefill.baseURL, "/health"))
			if err == nil {
				drainAndClose(response.Body)
				if response.StatusCode >= 200 && response.StatusCode < 300 {
					break
				}
			}
			time.Sleep(time.Second)
		}

		response, err := p.client.Get(prefill.bootstrapURL + "/query")
		if err != nil {
			log.Printf("Failed to query prefiller %s: %v", prefill.rawURL, err)
			return
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			drainAndClose(response.Body)
			log.Printf(
				"Failed to query prefiller %s: HTTP status %s",
				prefill.rawURL, response.Status,
			)
			return
		}

		var data map[string]struct {
			EngineID any `json:"engine_id"`
		}
		decoder := json.NewDecoder(response.Body)
		decoder.UseNumber()
		err = decoder.Decode(&data)
		response.Body.Close()
		if err != nil {
			log.Printf("Failed to decode prefiller query for %s: %v", prefill.rawURL, err)
			return
		}
		for rawRank, entry := range data {
			rank, err := strconv.Atoi(rawRank)
			if err != nil {
				log.Printf("Invalid DP rank %q from prefiller %s", rawRank, prefill.rawURL)
				return
			}
			prefill.dpEngineIDs[rank] = entry.EngineID
		}
		for rank := 0; rank < len(data); rank++ {
			if _, ok := prefill.dpEngineIDs[rank]; !ok {
				log.Printf("Missing DP rank %d from prefiller %s", rank, prefill.rawURL)
				return
			}
			p.prefillTargets = append(p.prefillTargets, prefillTarget{
				client: prefill,
				dpRank: rank,
			})
		}
		log.Printf("Inited prefiller %s with dp_size=%d", prefill.rawURL, len(data))
	}

	p.ready.Store(true)
	log.Print("All prefiller instances are ready.")
}

func (p *proxy) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	arrivedAt := time.Now()
	_, isCompletionEndpoint := completionEndpoints[request.URL.Path]
	isCompletionRequest := request.Method == http.MethodPost && isCompletionEndpoint
	requestID := request.Header.Get("X-Request-Id")
	var requestIDError error
	if isCompletionRequest {
		requestID, requestIDError = newUUID()
	}
	if requestID == "" {
		requestID = "-"
	}

	accessWriter := &accessLogResponseWriter{ResponseWriter: writer}
	log.Printf(
		"Request arrived: arrived_at=%s request_id=%q method=%s path=%q remote_addr=%q",
		arrivedAt.Format(time.RFC3339Nano), requestID, request.Method,
		request.URL.Path, request.RemoteAddr,
	)
	defer func() {
		completedAt := time.Now()
		statusCode := accessWriter.statusCode
		if statusCode == 0 {
			statusCode = http.StatusOK
		}
		log.Printf(
			"Request completed: completed_at=%s request_id=%q method=%s path=%q status=%d bytes=%d duration=%s",
			completedAt.Format(time.RFC3339Nano), requestID, request.Method,
			request.URL.Path, statusCode, accessWriter.bytesWritten,
			completedAt.Sub(arrivedAt),
		)
	}()

	if requestIDError != nil {
		log.Printf("Failed to generate request ID: %v", requestIDError)
		writeInternalError(accessWriter)
		return
	}
	if isCompletionRequest {
		p.handleCompletion(accessWriter, request, requestID)
		return
	}
	if len(p.decodeURLs) == 1 {
		p.proxyToSingleDecoder(accessWriter, request)
		return
	}
	writeJSONError(accessWriter, http.StatusNotFound, "Not Found")
}

func (p *proxy) proxyToSingleDecoder(
	writer http.ResponseWriter, request *http.Request,
) {
	targetURL := serviceURL(p.decodeURLs[0], request.URL.Path)
	if request.URL.RawQuery != "" {
		targetURL += "?" + request.URL.RawQuery
	}
	upstreamRequest, err := http.NewRequestWithContext(
		request.Context(), request.Method, targetURL, request.Body,
	)
	if err != nil {
		writeInternalError(writer)
		return
	}
	copyHeaders(upstreamRequest.Header, request.Header, true)
	upstreamRequest.ContentLength = request.ContentLength

	response, err := p.client.Do(upstreamRequest)
	if err != nil {
		log.Printf("Decoder proxy request failed: %v", err)
		writeInternalError(writer)
		return
	}
	defer response.Body.Close()

	copyHeaders(writer.Header(), response.Header, false)
	writer.WriteHeader(response.StatusCode)
	if err := copyStream(writer, response.Body); err != nil {
		log.Printf("Decoder proxy response failed: %v", err)
	}
}

func (p *proxy) handleCompletion(
	writer http.ResponseWriter, request *http.Request, requestID string,
) {
	if !p.ready.Load() {
		writeJSONError(writer, http.StatusServiceUnavailable, "Service Unavailable")
		return
	}

	requestData, err := decodeJSONObject(request.Body)
	if err != nil {
		log.Printf("Failed to decode completion request: %v", err)
		writeInternalError(writer)
		return
	}
	prefillTarget := p.nextPrefillTarget()
	decodeURL := p.nextDecodeURL()
	prefillBody, err := makePrefillBody(
		requestData, requestID, p.prefillDecodeMode,
	)
	if err != nil {
		log.Printf("Failed to encode prefill request %s: %v", requestID, err)
		writeInternalError(writer)
		return
	}

	var prefillKVTransferParams any
	if p.prefillDecodeMode == sequentialMode {
		prefillKVTransferParams, err = p.sendPrefill(
			request.Context(), prefillTarget, request.URL.Path,
			prefillBody, requestID,
		)
		if err != nil {
			log.Printf("Prefill request %s failed: %v", requestID, err)
			writeInternalError(writer)
			return
		}
	} else {
		go func() {
			if _, err := p.sendPrefill(
				context.Background(), prefillTarget, request.URL.Path,
				prefillBody, requestID,
			); err != nil {
				log.Printf("Concurrent prefill request %s failed: %v", requestID, err)
			}
		}()
	}

	var decodeBody []byte
	if p.prefillDecodeMode == sequentialMode {
		requestData["kv_transfer_params"] = prefillKVTransferParams
		decodeBody, err = json.Marshal(requestData)
	} else {
		decodeBody, err = makeDecodeBody(requestData, prefillTarget, requestID)
	}
	if err != nil {
		log.Printf("Failed to encode decode request %s: %v", requestID, err)
		writeInternalError(writer)
		return
	}
	streamStarted, err := p.streamDecode(
		writer, request.Context(), decodeURL, request.URL.Path,
		decodeBody, requestID,
	)
	if err != nil {
		log.Printf("Decode request %s failed: %v", requestID, err)
		if !streamStarted {
			writeInternalError(writer)
		}
	}
}

func (p *proxy) nextPrefillTarget() prefillTarget {
	index := p.prefillIndex.Add(1) - 1
	return p.prefillTargets[index%uint64(len(p.prefillTargets))]
}

func (p *proxy) nextDecodeURL() *url.URL {
	index := p.decodeIndex.Add(1) - 1
	return p.decodeURLs[index%uint64(len(p.decodeURLs))]
}

func makePrefillBody(
	requestData map[string]any, requestID, prefillDecodeMode string,
) ([]byte, error) {
	prefillData := cloneMap(requestData)
	kvTransferParams := map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
	}
	if prefillDecodeMode == concurrentMode {
		kvTransferParams["transfer_id"] = "xfer-" + requestID
	}
	prefillData["kv_transfer_params"] = kvTransferParams
	prefillData["stream"] = false
	prefillData["max_tokens"] = 1
	if _, ok := prefillData["max_completion_tokens"]; ok {
		prefillData["max_completion_tokens"] = 1
	}
	delete(prefillData, "stream_options")
	return json.Marshal(prefillData)
}

func makeDecodeBody(
	requestData map[string]any, target prefillTarget, requestID string,
) ([]byte, error) {
	requestData["kv_transfer_params"] = map[string]any{
		"do_remote_decode":      false,
		"do_remote_prefill":     true,
		"remote_bootstrap_addr": target.client.bootstrapURL,
		"remote_engine_id":      target.client.dpEngineIDs[target.dpRank],
		"transfer_id":           "xfer-" + requestID,
	}
	return json.Marshal(requestData)
}

func (p *proxy) sendPrefill(
	ctx context.Context,
	target prefillTarget,
	endpoint string,
	body []byte,
	requestID string,
) (any, error) {
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, serviceURL(target.client.baseURL, endpoint),
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", authorizationHeader())
	request.Header.Set("X-Request-Id", requestID)
	if p.prefillDecodeMode == concurrentMode {
		request.Header.Set("X-data-parallel-rank", strconv.Itoa(target.dpRank))
	}

	response, err := p.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		drainAndClose(response.Body)
		return nil, fmt.Errorf("HTTP status %s", response.Status)
	}
	if p.prefillDecodeMode == concurrentMode {
		_, err = io.Copy(io.Discard, response.Body)
		return nil, err
	}

	responseData, err := decodeJSONObject(response.Body)
	if err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	kvTransferParams, ok := responseData["kv_transfer_params"]
	if !ok {
		return nil, errors.New("prefill response missing kv_transfer_params")
	}
	return kvTransferParams, nil
}

func (p *proxy) streamDecode(
	writer http.ResponseWriter,
	ctx context.Context,
	decodeURL *url.URL,
	endpoint string,
	body []byte,
	requestID string,
) (bool, error) {
	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, serviceURL(decodeURL, endpoint),
		bytes.NewReader(body),
	)
	if err != nil {
		return false, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", authorizationHeader())
	request.Header.Set("X-Request-Id", requestID)

	response, err := p.client.Do(request)
	if err != nil {
		return false, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		drainAndClose(response.Body)
		return false, fmt.Errorf("HTTP status %s", response.Status)
	}

	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(http.StatusOK)
	return true, copyStream(writer, response.Body)
}

func decodeJSONObject(reader io.Reader) (map[string]any, error) {
	decoder := json.NewDecoder(reader)
	decoder.UseNumber()
	var data map[string]any
	if err := decoder.Decode(&data); err != nil {
		return nil, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("request contains more than one JSON value")
		}
		return nil, err
	}
	if data == nil {
		return nil, errors.New("request body must be a JSON object")
	}
	return data, nil
}

func cloneMap(source map[string]any) map[string]any {
	clone := make(map[string]any, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}

func serviceURL(base *url.URL, endpoint string) string {
	target := *base
	target.Path = strings.TrimRight(base.Path, "/") + "/" +
		strings.TrimLeft(endpoint, "/")
	target.RawPath = ""
	return target.String()
}

func authorizationHeader() string {
	apiKey, ok := os.LookupEnv("OPENAI_API_KEY")
	if !ok {
		apiKey = "None"
	}
	return "Bearer " + apiKey
}

func copyHeaders(destination, source http.Header, skipHost bool) {
	for name, values := range source {
		lowerName := strings.ToLower(name)
		if _, skip := hopByHopHeaders[lowerName]; skip || (skipHost && lowerName == "host") {
			continue
		}
		for _, value := range values {
			destination.Add(name, value)
		}
	}
}

func copyStream(writer http.ResponseWriter, reader io.Reader) error {
	bufferPointer := streamBufferPool.Get().(*[]byte)
	defer streamBufferPool.Put(bufferPointer)
	buffer := *bufferPointer
	flusher, canFlush := writer.(http.Flusher)

	for {
		read, readErr := reader.Read(buffer)
		if read > 0 {
			if _, err := writer.Write(buffer[:read]); err != nil {
				return err
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return nil
			}
			return readErr
		}
	}
}

func drainAndClose(body io.ReadCloser) {
	_, _ = io.Copy(io.Discard, body)
	_ = body.Close()
}

func writeJSONError(writer http.ResponseWriter, status int, detail string) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = fmt.Fprintf(writer, "{\"detail\":%q}", detail)
}

func writeInternalError(writer http.ResponseWriter) {
	http.Error(writer, "Internal Server Error", http.StatusInternalServerError)
}

func newUUID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	value[6] = (value[6] & 0x0f) | 0x40
	value[8] = (value[8] & 0x3f) | 0x80
	return fmt.Sprintf(
		"%x-%x-%x-%x-%x",
		value[0:4], value[4:6], value[6:8], value[8:10], value[10:16],
	), nil
}
