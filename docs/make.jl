using Documenter
using DocumenterVitepress    # VitePress frontend; native Mermaid via VitePress
using Literate               # .jl tutorial source → .md (with executed output)
using AtmosTransport

# ---------------------------------------------------------------------------
# Literate.jl preprocessing — every .jl file under docs/literate/ becomes a
# Markdown page under docs/src/tutorials/. Source is the canonical file;
# the generated .md files are rebuilt from scratch each Documenter run, so
# they live under docs/src/tutorials/_generated/ which is git-ignored.
# ---------------------------------------------------------------------------

const LITERATE_SRC = joinpath(@__DIR__, "literate")
const LITERATE_OUT = joinpath(@__DIR__, "src", "tutorials", "_generated")

if isdir(LITERATE_SRC)
    rm(LITERATE_OUT; force = true, recursive = true)
    mkpath(LITERATE_OUT)
    for jl in readdir(LITERATE_SRC; join = true)
        endswith(jl, ".jl") || continue
        Literate.markdown(jl, LITERATE_OUT;
                          flavor  = Literate.DocumenterFlavor(),
                          execute = true,
                          credit  = false)
    end
end

DocMeta.setdocmeta!(AtmosTransport, :DocTestSetup, :(using AtmosTransport);
                    recursive = true)

const PAGES = [
    "Home" => "index.md",
    "Learn" => [
        "Start here" => [
            "getting_started/installation.md",
            "getting_started/julia_basics.md",
            "getting_started/quickstart.md",
            "getting_started/first_run.md",
            "getting_started/inspecting_output.md",
            "getting_started/adjoints.md",
        ],
        "Core concepts" => [
            "concepts/architecture.md",
            "concepts/grids.md",
            "concepts/state_and_basis.md",
            "concepts/operators.md",
            "concepts/binary_format.md",
        ],
        "Tutorials" => [
            "tutorials/_generated/synthetic_latlon.md",
            "tutorials/_generated/first_adjoint.md",
        ],
    ],
    "Workflows" => [
        "Configuration and runtime" => [
            "config/toml_schema.md",
            "config/output_schema.md",
            "config/data_sources.md",
            "config/environment.md",
        ],
        "Meteorology preprocessing" => [
            "preprocessing/overview.md",
            "preprocessing/unified_binary_generation.md",
            "preprocessing/spectral_era5.md",
            "preprocessing/geos_native_cs.md",
            "preprocessing/regridding.md",
            "preprocessing/conventions.md",
        ],
    ],
    "TM5 & GCHP" => [
        "for_tm5_gchp_users/philosophy.md",
        "for_tm5_gchp_users/binary_pipeline.md",
        "for_tm5_gchp_users/operators_on_binaries.md",
        "for_tm5_gchp_users/adjoints.md",
        "for_tm5_gchp_users/kernel_architecture.md",
    ],
    "Theory" => [
        "theory/mass_conservation.md",
        "theory/advection_schemes.md",
        "theory/vertical_transport.md",
        "theory/conservation_budgets.md",
        "theory/float32_conservation.md",
        "theory/validation_status.md",
        "theory/adjoint_status.md",
    ],
    "API" => [
        "api/public_api.md",
        "api/index.md",
        "api/architectures.md",
        "api/parameters.md",
        "api/grids.md",
        "api/state.md",
        "api/met_drivers.md",
        "api/operators.md",
        "api/models.md",
        "api/preprocessing.md",
        "api/downloads.md",
        "api/regridding.md",
        "api/output_visualization.md",
        "api/adjoints.md",
        "api/infrastructure.md",
    ],
    "About these docs" => "about.md",
]

makedocs(
    root     = @__DIR__,
    modules  = [AtmosTransport],
    sitename = "AtmosTransport.jl",
    authors  = "RemoteSensingTools and contributors",
    repo     = Remotes.GitHub("RemoteSensingTools", "AtmosTransport.jl"),
    format   = DocumenterVitepress.MarkdownVitepress(
        repo          = "github.com/RemoteSensingTools/AtmosTransport.jl",
        devbranch     = "main",
        devurl        = "dev",
    ),
    pages    = PAGES,
    # Broken doctests/references and unpublished exported docstrings are
    # release-blocking documentation defects.
    warnonly = false,
    checkdocs = :exports,
)

# VitePress front matter must be the first bytes of the converted Markdown.
# A preceding Documenter-only block produces a blank line, causing VitePress to
# render the YAML as page content instead of selecting its home-page layout.
const CONVERTED_HOME = joinpath(@__DIR__, "build", ".documenter", "index.md")
const converted_home = read(CONVERTED_HOME, String)
startswith(converted_home, "---\nlayout: home\n") ||
    error("converted home page does not begin with VitePress front matter")

const rendered_homes = filter(readdir(joinpath(@__DIR__, "build"); join = true)) do path
    isfile(joinpath(path, "index.html"))
end
isempty(rendered_homes) && error("VitePress produced no rendered home page")
for path in rendered_homes
    rendered_home = read(joinpath(path, "index.html"), String)
    isfile(joinpath(path, "logo.png")) ||
        error("rendered home page is missing its public logo asset")
    occursin("VPHome", rendered_home) ||
        error("rendered home page did not select the VitePress home layout")
    occursin("<p>layout: home</p>", rendered_home) &&
        error("rendered home page contains unparsed VitePress front matter")
end

# DocumenterVitepress builds separate VitePress outputs under docs/build/1,
# docs/build/2, ... and records their destination folders in docs/build/bases.txt.
# Its deploy wrapper publishes those rendered folders into gh-pages/dev,
# gh-pages/stable, etc. Calling Documenter.deploydocs directly would publish the
# wrapper directory itself and leave the public /dev/ URL without an index page.
const ATMOSTR_DOCS_BUILD_ONLY =
    lowercase(get(ENV, "ATMOSTR_DOCS_BUILD_ONLY", "false")) in ("1", "true", "yes")

if ATMOSTR_DOCS_BUILD_ONLY
    @info "Skipping docs deployment because ATMOSTR_DOCS_BUILD_ONLY is set"
else
    DocumenterVitepress.deploydocs(
        root         = @__DIR__,
        repo         = "github.com/RemoteSensingTools/AtmosTransport.jl.git",
        target       = "build",
        devbranch    = "main",
        branch       = "gh-pages",
        push_preview = true,
        forcepush    = true,
    )
end
