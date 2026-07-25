package fibre

import (
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/stretchr/testify/require"
)

// TestNewBlobFromExtendedData asserts the core invariant: wrapping an
// already-encoded rsema1d GKR square yields a blob whose ID commitment is
// byte-identical to the square's own commitment (no re-encode, no header).
func TestNewBlobFromExtendedData(t *testing.T) {
	for _, numVars := range []int{4, 6, 11, 13} {
		numVars := numVars
		t.Run("numVars="+itoa(numVars), func(t *testing.T) {
			inputVals := make([]byte, 1<<uint(numVars))
			for i := range inputVals {
				inputVals[i] = byte((i*31 + 7) & 0xFF)
			}

			ed, err := rsema1d.EncodeGKRInputSquare(inputVals, numVars)
			require.NoError(t, err)

			blob, err := NewBlobFromExtendedData(ed)
			require.NoError(t, err)

			// The whole point: commitment is preserved exactly.
			require.Equal(t, Commitment(ed.Commitment()), blob.ID().Commitment(),
				"blob commitment must equal the rsema1d square commitment")
			require.Equal(t, uint8(1), blob.ID().Version())

			// Derived config must match the square's real K/N/rowLen.
			k := 1 << uint(numVars-2)
			require.Equal(t, k, blob.Config().OriginalRows, "K")
			require.Equal(t, k, blob.Config().ParityRows, "N (GKR square is K=N)")
			require.Equal(t, 2*k, blob.Config().TotalRows())
			require.Equal(t, len(ed.Row(0)), blob.RowSize())
			require.Equal(t, len(ed.Row(0))*k, blob.UploadSize())
			require.Equal(t, len(ed.RLC()), k)
		})
	}
}

// itoa is a tiny local int->string helper to avoid importing strconv just for
// subtest names.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
