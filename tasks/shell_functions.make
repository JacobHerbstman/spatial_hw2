JULIA = @source ../../shell_functions.sh; Julia_pc_and_local

# If 'make -n' option is invoked
ifneq (,$(findstring n,$(MAKEFLAGS)))
JULIA := julia
endif
