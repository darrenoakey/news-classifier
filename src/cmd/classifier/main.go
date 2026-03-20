// Command classifier is the news classifier HTTP service.
// It loads exported sklearn models from a directory and serves predictions.
//
// Usage:
//
//	classifier -models local/models -addr :8810
//
// Endpoints:
//
//	POST /classify {"title":"..."}        → tree/svm label + score
//	POST /execute  engine envelope format → classification + passthrough fields
//	GET  /health                          → {"ok":true}
package main

import (
	"flag"
	"fmt"
	"log"
	"news-classifier/pkg/classifier"
	"os"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	modelDir := flag.String("models", "local/models", "directory containing tree_model.json and svm_model.json")
	addr := flag.String("addr", ":8810", "HTTP listen address")
	flag.Parse()

	log.Printf("loading models from %s", *modelDir)
	c, err := classifier.New(*modelDir)
	if err != nil {
		return fmt.Errorf("load models: %w", err)
	}
	log.Printf("models loaded")

	srv := classifier.NewServer(c, *addr)
	return srv.ListenAndServe()
}
