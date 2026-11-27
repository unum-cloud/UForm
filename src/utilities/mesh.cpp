/**
 *  @file src/utilities/mesh.cpp
 *  @author Ash Vardanian
 *  @date July 2, 2026
 *  @brief The world bootstrap's text exchange: parsing the facts and manifest payloads and the
 *      lead's agreement checks, over StringZilla's verbs — see the header for the contract.
 */
#include <cstdlib> // `std::getenv`
#include <cstring> // `std::memcmp`, `std::strcmp`

#include <algorithm> // `std::sort` — the world-GPU-uniqueness check
#include <utility>   // `std::move`

#include <stringzilla/find.hpp> // `sz::split_characters`, `sz::partition`, `sz::strip`, `sz::contains`

#include "uform/utilities/mesh.hpp"

namespace util = unum::uform::utilities;

util::expected<util::mesh_topology_t> util::mesh_topology_t::bootstrap(
    u32_t host_domain_index, u32_t host_domain_count, mesh_pause_t poll_pause, u32_t gpus_per_multimem_domain,
    sz::string_view_t directory, sz::string_view_t job_id, f64_t timeout_seconds, u32_t gpu_class) noexcept {
    if (host_domain_count == 0 || host_domain_index >= host_domain_count)
        return expected<mesh_topology_t>::fail( //
            status_code_t::invalid_argument_k,
            "host_domain_index must be in [0,host_domain_count) (host_domain_index={} host_domain_count={})",
            host_domain_index, host_domain_count);
    auto [local, discover_status] = discover_mesh_shape();
    if (!discover_status) return expected<mesh_topology_t>::fail(std::move(discover_status));
    u32_t const local_devices = local.gpus_per_host_domain;

    // The multimem extent is explicit configuration (see the parameter docs); the
    // nesting invariants are enforced by the same make_mesh_shape every hand-configured
    // world goes through.
    if (gpus_per_multimem_domain == 0)
        if (char const* value = std::getenv("UFORM_GPUS_PER_MULTIMEM_DOMAIN"); value != nullptr) // vendor-boundary
            if (!parse_u32_(sz::string_view_t {value}, gpus_per_multimem_domain))
                return expected<mesh_topology_t>::fail(status_code_t::invalid_argument_k,
                                                       "$UFORM_GPUS_PER_MULTIMEM_DOMAIN is not a number: \"{}\"",
                                                       value);
    if (gpus_per_multimem_domain == 0) gpus_per_multimem_domain = local_devices;
    auto [shape, make_status] = make_mesh_shape(local_devices, gpus_per_multimem_domain,
                                                local_devices * host_domain_count);
    if (!make_status) return expected<mesh_topology_t>::fail(std::move(make_status));

    auto [local_probe, probe_status] = probe_local_host_domain_(host_domain_index, local_devices);
    if (!probe_status) return expected<mesh_topology_t>::fail(std::move(probe_status));

    mesh_topology_t topology;
    topology.shape = shape;
    topology.host_domain_index = host_domain_index;
    topology.poll_pause = poll_pause;
    if (st::failed(topology.host_domains.resize(host_domain_count)))
        return expected<mesh_topology_t>::fail(
            status_code_t::out_of_memory_k, "host_domain table allocation failed (host_domains={})", host_domain_count);
    topology.host_domains[host_domain_index] = local_probe.facts;
    if (expected_t pinned = topology.pin_faults_(); !pinned) return expected<mesh_topology_t>::fail(std::move(pinned));
    if (host_domain_count == 1) return expected<mesh_topology_t> {std::move(topology)};

    // The exchange root and the job ID are explicit configuration; guessing either is how worlds
    // hang (an unshared directory) or how node families collide (an anonymous namespace).
    sz::string_view_t base = directory, job = job_id;
    if (base.empty())
        if (char const* value = std::getenv("UFORM_EXCHANGE_DIR"); value != nullptr) // vendor-boundary
            base = sz::string_view_t {value};
    if (base.empty())
        return expected<mesh_topology_t>::fail( //
            status_code_t::invalid_argument_k,
            "a multi-host_domain bootstrap needs a shared exchange directory: pass one or set " //
            "$UFORM_EXCHANGE_DIR (host_domain_count={})",
            host_domain_count);
    if (job.empty())
        if (char const* value = std::getenv("UFORM_JOB_ID"); value != nullptr) // vendor-boundary
            job = sz::string_view_t {value};
    if (job.empty())
        return expected<mesh_topology_t>::fail( //
            status_code_t::invalid_argument_k,
            "a multi-host_domain bootstrap needs a unique job id: pass one or set $UFORM_JOB_ID " //
            "(host_domain_count={})",
            host_domain_count);
    if (sz::contains(base, '\0') || sz::contains(job, '\0'))
        return expected<mesh_topology_t>::fail(status_code_t::invalid_argument_k, "Directory and job id hold no NUL");
    if (!format_into(topology.exchange_directory, "{}/{}", base, job))
        return expected<mesh_topology_t>::fail(status_code_t::out_of_memory_k, "Exchange path assembly failed");
    if (int const directory_error = make_directories_(topology.exchange_directory.view()); directory_error != 0)
        return expected<mesh_topology_t>::fail(status_code_t::io_k,
                                               "creating the exchange directory failed (path={} error={})",
                                               topology.exchange_directory, directory_error);

    sz::string_t facts_text;
    if (!build_facts_(facts_text, local_probe.facts, topology.shape, job, gpu_class, local_probe.uuids.view()))
        return expected<mesh_topology_t>::fail(status_code_t::out_of_memory_k, "facts assembly failed (host_domain={})",
                                               host_domain_index);
    if (expected_t published = topology.publish_text("mesh", facts_text.view()); !published)
        return expected<mesh_topology_t>::fail(std::move(published));

    expected_t exchanged = host_domain_index == 0
                               ? topology.collect_world_as_lead_(job, local_probe, gpu_class, timeout_seconds)
                               : topology.adopt_manifest_(job, local_probe.facts, timeout_seconds);
    if (!exchanged) return expected<mesh_topology_t>::fail(std::move(exchanged));

    if (expected_t synchronized = topology.barrier("mesh", timeout_seconds); !synchronized)
        return expected<mesh_topology_t>::fail(std::move(synchronized));
    return expected<mesh_topology_t> {std::move(topology)};
}

bool util::mesh_topology_t::parse_facts_(sz::string_view_t payload, facts_t& facts) noexcept {
    sz::byteset_t const whitespaces = sz::whitespaces_set();
    bool index_parsed = false, count_parsed = false, multimem_parsed = false, devices_parsed = false,
         class_parsed = false;
    for (sz::string_view_t line : sz::split_characters(payload, sz::newlines_set())) {
        auto [key, match, value] = sz::partition(line, ' ');
        value = sz::strip(value, whitespaces);
        if (key == "schema") facts.schema = value;
        else if (key == "job") facts.job = value;
        else if (key == "hostname") facts.hostname = value;
        else if (key == "device_name") facts.device_name = value;
        else if (key == "uuids") facts.uuids = value;
        else if (key == "host_domain_index") index_parsed = parse_u32_(value, facts.host_domain_index);
        else if (key == "host_domain_count") count_parsed = parse_u32_(value, facts.host_domain_count);
        else if (key == "multimem") multimem_parsed = parse_u32_(value, facts.multimem);
        else if (key == "gpu_class") class_parsed = parse_u32_(value, facts.gpu_class);
        else if (key == "devices") devices_parsed = parse_u32_(value, facts.devices);
    }
    return index_parsed && count_parsed && multimem_parsed && class_parsed && devices_parsed && !facts.schema.empty() &&
           !facts.job.empty() && !facts.hostname.empty() && !facts.device_name.empty() && !facts.uuids.empty();
}

bool util::mesh_topology_t::parse_manifest_(sz::string_view_t payload, sz::string_view_t job,
                                            u32_t gpus_per_multimem_domain,
                                            st::basic_vector<mesh_host_domain_t>& table) noexcept {
    auto take_token = [](sz::string_view_t& row) noexcept {
        // Pop the leading space-separated token off the row with the partition API.
        auto [token, match, rest] = sz::partition(row, ' ');
        row = rest;
        return token;
    };
    auto parse_row = [&take_token](sz::string_view_t row, mesh_host_domain_t& entry) noexcept {
        if (take_token(row) != "host_domain" || !parse_u32_(take_token(row), entry.host_domain_index)) return false;
        if (take_token(row) != "hostname") return false;
        copy_token_(take_token(row), entry.hostname);
        if (take_token(row) != "devices" || !parse_u32_(take_token(row), entry.device_count)) return false;
        if (take_token(row) != "device_name") return false;
        copy_token_(take_token(row), entry.device_name);
        return entry.hostname[0] != '\0' && entry.device_name[0] != '\0';
    };

    sz::byteset_t const whitespaces = sz::whitespaces_set();
    std::size_t parsed_rows = 0;
    u32_t multimem = 0;
    bool schema_matched = false, job_matched = false, multimem_matched = false;
    for (sz::string_view_t line : sz::split_characters(payload, sz::newlines_set())) {
        auto [key, match, value] = sz::partition(line, ' ');
        if (key == "schema") schema_matched = sz::strip(value, whitespaces) == mesh_schema_k;
        else if (key == "job") job_matched = sz::strip(value, whitespaces) == job;
        else if (key == "multimem")
            multimem_matched = parse_u32_(sz::strip(value, whitespaces), multimem) &&
                               multimem == gpus_per_multimem_domain;
        else if (key == "host_domain") {
            mesh_host_domain_t entry;
            if (!parse_row(line, entry) || entry.host_domain_index >= table.size()) return false;
            table[entry.host_domain_index] = entry;
            ++parsed_rows;
        }
    }
    return schema_matched && job_matched && multimem_matched && parsed_rows == table.size();
}

util::expected_t util::mesh_topology_t::collect_world_as_lead_(sz::string_view_t job, local_probe_t const& probed,
                                                               u32_t self_gpu_class, f64_t timeout_seconds) noexcept {
    mesh_host_domain_t const& self = probed.facts;
    st::basic_vector<claimed_gpu_t> claimed;
    // The world's physical GPU identities, this host_domain's first; sorted once at the end to
    // prove none is claimed twice — O(world log world), never a pairwise scan.
    if (st::failed(claimed.reserve(shape.gpus_per_colocated_domain)))
        return expected_t::fail(status_code_t::out_of_memory_k, "GPU-uniqueness table allocation failed ({} GPUs)",
                                shape.gpus_per_colocated_domain);
    for (CUuuid const& uuid : probed.uuids.view())
        if (st::failed(claimed.push_back({uuid, 0u})))
            return expected_t::fail(status_code_t::out_of_memory_k, "GPU-uniqueness table push failed");

    for (u32_t peer_index = 1; peer_index != host_domain_count(); ++peer_index) {
        sz::string_t peer_facts;
        if (expected_t awaited = await_text(peer_index, "mesh", peer_facts, timeout_seconds); !awaited) return awaited;
        facts_t facts;
        if (!parse_facts_(peer_facts.view(), facts))
            return expected_t::fail(status_code_t::io_k, "malformed facts from host_domain {}: \"{}\"", peer_index,
                                    peer_facts);
        if (facts.schema != sz::string_view_t {mesh_schema_k})
            return expected_t::fail(status_code_t::io_k, "host_domain {} published schema \"{}\", expected {}",
                                    peer_index, facts.schema, mesh_schema_k);
        if (facts.job != job)
            return expected_t::fail(status_code_t::io_k, "host_domain {} belongs to job \"{}\", expected \"{}\"",
                                    peer_index, facts.job, job);
        if (facts.host_domain_index != peer_index)
            return expected_t::fail(status_code_t::io_k, "host_domain {} published a mismatched index {}", peer_index,
                                    facts.host_domain_index);
        if (facts.host_domain_count != host_domain_count())
            return expected_t::fail( //
                status_code_t::unsupported_k,
                "world-size disagreement: host_domain {} was launched with {} host_domains, host_domain 0 with {}",
                peer_index, facts.host_domain_count, host_domain_count());
        if (facts.multimem != shape.gpus_per_multimem_domain)
            return expected_t::fail( //
                status_code_t::unsupported_k,
                "multimem-extent disagreement: host_domain {} was launched with {}, host_domain 0 with {}", peer_index,
                facts.multimem, shape.gpus_per_multimem_domain);
        if (facts.gpu_class != self_gpu_class)
            return expected_t::fail( //
                status_code_t::unsupported_k,
                "GPU class disagreement: host_domain {} trains as class {}, host_domain 0 as class {}", peer_index,
                facts.gpu_class, self_gpu_class);
        mesh_host_domain_t peer;
        peer.host_domain_index = peer_index;
        peer.device_count = facts.devices;
        copy_token_(facts.hostname, peer.hostname);
        copy_token_(facts.device_name, peer.device_name);
        if (peer.device_count != self.device_count || std::strcmp(peer.device_name, self.device_name) != 0)
            return expected_t::fail( //
                status_code_t::unsupported_k,
                "the mesh must be uniform: host_domain {} ({}) has {}×{}, host_domain 0 ({}) has {}×{}", peer_index,
                peer.hostname, peer.device_count, peer.device_name, self.hostname, self.device_count, self.device_name);
        std::size_t peer_uuids = 0;
        // The peer's physical GPU identities join the world table; one uuid per claimed device.
        for (sz::string_view_t token : sz::split_characters(facts.uuids, sz::whitespaces_set())) {
            if (token.empty()) continue;
            CUuuid uuid {};
            if (!parse_uuid_(token, uuid))
                return expected_t::fail(status_code_t::io_k, "host_domain {} published a malformed GPU uuid \"{}\"",
                                        peer_index, token);
            if (st::failed(claimed.push_back({uuid, peer_index})))
                return expected_t::fail(status_code_t::out_of_memory_k, "GPU-uniqueness table push failed");
            ++peer_uuids;
        }
        if (peer_uuids != facts.devices)
            return expected_t::fail(status_code_t::io_k, "host_domain {} listed {} GPU uuids for {} devices",
                                    peer_index, peer_uuids, facts.devices);
        host_domains[peer_index] = peer;
    }

    auto const by_uuid = [](claimed_gpu_t const& left, claimed_gpu_t const& right) noexcept {
        return std::memcmp(left.uuid.bytes, right.uuid.bytes, sizeof(left.uuid.bytes)) < 0;
    };
    // No physical GPU is owned by two host_domains — the guard against an unpartitioned
    // CUDA_VISIBLE_DEVICES (co-located processes seeing the same devices). One sort,
    // one adjacent scan.
    std::sort(claimed.data(), claimed.data() + claimed.size(), by_uuid);
    for (std::size_t index = 1; index < claimed.size(); ++index)
        if (std::memcmp(claimed[index].uuid.bytes, claimed[index - 1u].uuid.bytes, sizeof(claimed[index].uuid.bytes)) ==
            0) {
            sz::string_t uuid_hex;
            [[maybe_unused]] bool const formatted = format_uuid_(uuid_hex, claimed[index].uuid);
            return expected_t::fail(
                status_code_t::unsupported_k,
                "GPU {} is claimed by host_domain {} and {}; the launcher left CUDA_VISIBLE_DEVICES unpartitioned",
                uuid_hex, claimed[index - 1u].owner, claimed[index].owner);
        }

    sz::string_t manifest;
    if (!build_manifest_(manifest, host_domains, shape, job))
        return expected_t::fail(status_code_t::out_of_memory_k, "manifest assembly failed (rows={})",
                                host_domain_count());
    return publish_text("manifest", manifest.view());
}
