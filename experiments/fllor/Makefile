BASE_DIR=experiments/fllor
OUTPUT_DIR=$(BASE_DIR)/outputs
PLOT_DIR=$(BASE_DIR)/figures

ENVS=bandit newcomb damascus asymmetric-damascus coordination pdbandit switching
AGENTS=classical bayesian exp3 experimental1 experimental2
POLICIES=epsilon softmax
OPTIONS_BASE=--steps 1001 --runs 1000
OPTIONS_bandit=--arms 10
POLICY_epsilon=epsilon=0.5:0.5:0.01
POLICY_softmax=temperature=1:0.3:0.05

SOURCES=$(BASE_DIR)/main.py $(wildcard ibrl/*.py) $(wildcard ibrl/*/*.py)

all: $(filter-out $(PLOT_DIR)/bandit.experimental2.png,$(foreach env,$(ENVS),$(foreach agent,$(AGENTS),$(PLOT_DIR)/$(env).$(agent).png)))
.PHONY: all

$(OUTPUT_DIR) $(PLOT_DIR): %:
	mkdir -p $@

# This is a bit of a mess, because not some environment/agents/policy combinations need to be excluded or require special arguments

# Perform simulation. One target per (environment,agent,policy)
define CREATE_RUN_TARGET # env agent policy
$(OUTPUT_DIR)/$1.$2.$3.txt: $(SOURCES) | $(OUTPUT_DIR)
	uv run python -m $$(subst /,.,$$(basename $$<)) $1 $2:$$(POLICY_$3) $$(OPTIONS_BASE) $$(OPTIONS_$1) > $$@
endef
define CREATE_RUN_TARGET_2 # env agent policy
$(OUTPUT_DIR)/$1.$2.epsilon.txt: $(SOURCES) | $(OUTPUT_DIR)
	uv run python -m $$(subst /,.,$$(basename $$<)) $1 $2:epsilon=0.5:700:0.01,decay_type=1 $$(OPTIONS_BASE) $$(OPTIONS_$1) > $$@
endef
define CREATE_RUN_TARGET_3 # env agent policy
$(OUTPUT_DIR)/$1.$2.epsilon.txt: $(SOURCES) | $(OUTPUT_DIR)
	uv run python -m $$(subst /,.,$$(basename $$<)) $1 $2 $$(OPTIONS_BASE) $$(OPTIONS_$1) > $$@
endef
$(foreach env,$(ENVS),$(foreach agent,$(filter-out experimental2 exp3,$(AGENTS)),$(foreach policy,$(POLICIES),$(eval $(call CREATE_RUN_TARGET,$(env),$(agent),$(policy))))))
$(foreach env,$(ENVS),$(eval $(call CREATE_RUN_TARGET_2,$(env),experimental2,epsilon)))
$(foreach env,$(ENVS),$(eval $(call CREATE_RUN_TARGET_3,$(env),exp3,epsilon)))

# Create plot. One target per (environment,agent)
define CREATE_PLOT_TARGET # env agent
$(PLOT_DIR)/$1.$2.png: $(BASE_DIR)/plot.gp $(OUTPUT_DIR)/$1.$2.epsilon.txt $(OUTPUT_DIR)/$1.$2.softmax.txt | $(PLOT_DIR)
	gnuplot -c $$< $1 $2
endef
define CREATE_PLOT_TARGET_2 # env agent
$(PLOT_DIR)/$1.$2.png: $(BASE_DIR)/plot.gp $(OUTPUT_DIR)/$1.$2.epsilon.txt | $(PLOT_DIR)
	gnuplot -c $$< $1 $2
endef
$(foreach env,$(ENVS),$(foreach agent,$(filter-out experimental2 exp3,$(AGENTS)),$(eval $(call CREATE_PLOT_TARGET,$(env),$(agent)))))
$(foreach env,$(ENVS),$(foreach agent,experimental2 exp3,$(eval $(call CREATE_PLOT_TARGET_2,$(env),$(agent)))))

# Shortcuts to create all plots involving a specific environment or agent
.PHONY: $(ENVS) $(AGENTS)
$(filter-out experimental2,$(AGENTS)): %: $(foreach env,$(ENVS),$(PLOT_DIR)/$(env).%.png)
experimental2: %: $(foreach env,$(filter-out bandit,$(ENVS)),$(PLOT_DIR)/$(env).%.png)
$(filter-out bandit,$(ENVS)): %: $(foreach agent,$(AGENTS),$(PLOT_DIR)/%.$(agent).png)
bandit: %: $(foreach agent,$(filter-out experimental2,$(AGENTS)),$(PLOT_DIR)/%.$(agent).png)

.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR) $(PLOT_DIR)

