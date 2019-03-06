push!(LOAD_PATH,"../src/")
push!(LOAD_PATH,"../src/operators/")

using
    Documenter,
    StaggeredPoisson

makedocs(
   modules = [StaggeredPoisson],
   doctest = false,
   clean   = true,
 checkdocs = :all,
    assets = ["assets/invenia.css"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
   authors = "Somebody",
  sitename = "StaggeredPoisson.jl",
     pages = ["Home" => "index.md"]
)

deploydocs(repo = "github.com/glwagner/StaggeredPoisson.git")
