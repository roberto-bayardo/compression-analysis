CFLAGS="-I /Users/bayardo/src/github.com/google/brotli/c/include" 
LDFLAGS="-L /Users/bayardo/src/github.com/google/brotli -Wl,-rpath,/Users/bayardo/src/github.com/google/brotli"

main: Makefile *.go
	CGO_CFLAGS=$(CFLAGS) CGO_LDFLAGS=$(LDFLAGS) go build -o main *.go
