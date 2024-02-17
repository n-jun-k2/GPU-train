show/projects:
	@docker compose run -u root --rm q dotnet new -i Microsoft.Quantum.ProjectTemplates -n console

project/%:
	@docker compose run -u root --rm q dotnet new $(*D) -lang Q# -o $(@F)

run/%:
	@docker compose run -u root --rm q dotnet run --project $(@F) --configuration $(*D) $(ARGS)

clean/%:
	@docker compose run -u root --rm q dotnet clean $(@F) --configuration $(*D)

test/%:
	@docker compose run -u root --rm q /bin/bash -c "cd $(@F) && dotnet restore && dotnet test"

restore/%:
	@docker compose run -u root --rm q /bin/bash -c "cd $(@F) && dotnet restore"

add/package/%:
	docker compose run -u root --rm q dotnet add $(*D) package $(@F)

add/reference/%:
	docker compose run -u root --rm q dotnet add $(*D) reference $(@F)

clear:
	docker compose run -u root --rm q dotnet nuget locals all --clear

show/package/%:
	docker compose run -u root --rm q dotnet show package $(@F)

list/%:
	docker compose run -u root --rm q  dotnet list $(@F) package

remove/%:
	docker compose run -u root --rm q dotnet remove $(@F).csproj package Microsoft.Quantum.Development.Kit

pyrun/%:
	@docker compose run -u root --rm q bash -c "python $(@F)"

qbash:
	@docker compose run -u root --rm q bash 