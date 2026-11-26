// Tests of what the binding decides without the library: which arena descriptions Go code reads.
//
// File: golang/types_internal_test.go
// Author: Ash Vardanian
// Date: November 26, 2026

package uform

import (
	"errors"
	"testing"
)

// The stub makes only host arenas, so device descriptions are built by hand: host reads key on
// HostAccess alone, never on the residency.
func TestHostReadsFollowHostAccess(t *testing.T) {
	device := ArenaDescription{Residency: ResidencyVram, Device: 0, Vendor: GPUVendorNVIDIA, VendorOrdinal: 0, HostAccess: HostAccessRefused}
	if err := device.host(); !errors.Is(err, StatusDevice) {
		t.Fatalf("device memory read as host memory: %v", err)
	}
	shared := ArenaDescription{Residency: ResidencyVram, Device: 0, Vendor: GPUVendorApple, VendorOrdinal: 0, HostAccess: HostAccessReadable}
	if err := shared.host(); err != nil {
		t.Fatalf("Apple's shared memory refused: %v", err)
	}
}
