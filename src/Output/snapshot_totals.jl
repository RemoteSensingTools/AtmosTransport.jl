# Global tracer totals for snapshots: compensated Float64 sums
# (`Architectures.compensated_total`), independent of the storage precision
# and reduced on the device where the backend supports Float64.
function _backend_tracer_total(values)
    total = compensated_total(values)
    isfinite(total) || throw(ArgumentError(
        "snapshot tracer total is not finite; check the captured tracer state"))
    return total
end
