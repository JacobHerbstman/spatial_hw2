function _fill_din_k!(Din_k::Matrix{Float64}, Din::Matrix{Float64}, kappa_hat::Matrix{Float64},
                      inv_LT_mat::Matrix{Float64}; threaded::Bool = false,
                      kappa_is_one::Bool = false)
    if kappa_is_one
        Din_k .= Din
        return Din_k
    end
    if threaded && Threads.nthreads() > 1
        Threads.@threads for j in axes(Din_k, 2)
            @inbounds for i in axes(Din_k, 1)
                Din_k[i, j] = Din[i, j] * (kappa_hat[i, j] ^ inv_LT_mat[i, j])
            end
        end
    else
        @inbounds for j in axes(Din_k, 2), i in axes(Din_k, 1)
            Din_k[i, j] = Din[i, j] * (kappa_hat[i, j] ^ inv_LT_mat[i, j])
        end
    end
    Din_k
end

function _is_all_ones(A::AbstractArray{<:Real})
    @inbounds for x in A
        if x != 1.0
            return false
        end
    end
    true
end

function p_h_om!(ws::TempEqWorkspace, om::Matrix{Float64}, kappa_hat::Matrix{Float64},
                 lambda_hat::Matrix{Float64}, theta::Vector{Float64}, G::Matrix{Float64},
                 gamma::Matrix{Float64}, Din::Matrix{Float64};
                 maxit::Int = 2000, tol::Float64 = 1e-7, threaded::Bool = false,
                 kappa_is_one::Bool = false, lambda_is_one::Bool = false,
                 reset_price_guess::Bool = true)
    J, N = size(om)
    pf0 = ws.pf0
    phat = ws.phat
    c = ws.c
    lc = ws.lc
    lom = ws.lom
    lp = ws.lp
    Din_k = ws.Din_k
    tmp_row = ws.tmp_row

    if reset_price_guess
        fill!(pf0, 1.0)
    end
    fill!(phat, 1.0)
    fill!(c, 1.0)

    pfmax = 1.0
    it = 1

    while (it <= maxit) && (pfmax > tol)
        @inbounds for n in 1:N, j in 1:J
            lom[j, n] = log(om[j, n])
            lp[j, n] = log(pf0[j, n])
        end

        @inbounds for i in 1:N
            row0 = (i - 1) * J
            for j in 1:J
                acc = 0.0
                for jp in 1:J
                    acc += G[row0 + jp, j] * lp[jp, i]
                end
                lc[j, i] = gamma[j, i] * lom[j, i] + acc
                c[j, i] = exp(lc[j, i])
            end
        end

        _fill_din_k!(
            Din_k,
            Din,
            kappa_hat,
            ws.inv_LT_mat;
            threaded = threaded,
            kappa_is_one = kappa_is_one,
        )

        @inbounds for j in 1:J
            inv_theta = 1.0 / theta[j]
            neg_theta = -theta[j]
            for n in 1:N
                if lambda_is_one
                    tmp_row[n] = c[j, n] ^ (-inv_theta)
                else
                    tmp_row[n] = (lambda_hat[j, n] ^ (gamma[j, n] * inv_theta)) * (c[j, n] ^ (-inv_theta))
                end
            end
            for n in 1:N
                idx = n + (j - 1) * N
                dotv = 0.0
                for m in 1:N
                    dotv += Din_k[idx, m] * tmp_row[m]
                end
                phat[j, n] = dotv ^ neg_theta
            end
        end

        pfmax = 0.0
        @inbounds for idx in eachindex(phat)
            dev = abs(phat[idx] - pf0[idx])
            pf0[idx] = phat[idx]
            if dev > pfmax
                pfmax = dev
            end
        end
        it += 1
    end

    return pf0, c
end

function dinprime!(ws::TempEqWorkspace, Din::Matrix{Float64}, kappa_hat::Matrix{Float64},
                   lambda_hat::Matrix{Float64}, c::Matrix{Float64}, phat::Matrix{Float64},
                   theta::Vector{Float64}, gamma::Matrix{Float64}; threaded::Bool = false,
                   kappa_is_one::Bool = false, lambda_is_one::Bool = false)
    J, N = size(c)
    cp = ws.cp
    phatp = ws.phatp
    inner_term = ws.inner_term
    Dinp = ws.Dinp
    Din_k = ws.Din_k

    _fill_din_k!(
        Din_k,
        Din,
        kappa_hat,
        ws.inv_LT_mat;
        threaded = threaded,
        kappa_is_one = kappa_is_one,
    )

    @inbounds for j in 1:J
        inv_theta = 1.0 / theta[j]
        for n in 1:N
            cp[j, n] = c[j, n] ^ (-inv_theta)
            phatp[j, n] = phat[j, n] ^ (-inv_theta)
            if lambda_is_one
                inner_term[j, n] = cp[j, n]
            else
                inner_term[j, n] = cp[j, n] * (lambda_hat[j, n] ^ (gamma[j, n] * inv_theta))
            end
        end
    end

    if threaded && Threads.nthreads() > 1
        Threads.@threads for n in 1:N
            @inbounds for j in 1:J
                idx = n + (j - 1) * N
                inv_phat = 1.0 / phatp[j, n]
                for m in 1:N
                    Dinp[idx, m] = Din_k[idx, m] * inner_term[j, m] * inv_phat
                end
            end
        end
    else
        @inbounds for n in 1:N, j in 1:J
            idx = n + (j - 1) * N
            inv_phat = 1.0 / phatp[j, n]
            for m in 1:N
                Dinp[idx, m] = Din_k[idx, m] * inner_term[j, m] * inv_phat
            end
        end
    end

    Dinp
end

function expenditurenew!(ws::TempEqWorkspace, J::Int, N::Int, alphas::Matrix{Float64}, B::Matrix{Float64},
                         Dinp::Matrix{Float64}, om::Matrix{Float64}, Ljn_hat::Matrix{Float64},
                         Snp::Vector{Float64}, VARjn0::Matrix{Float64}, VALjn0::Matrix{Float64},
                         io::Vector{Float64})
    JN = J * N
    VARjnp = ws.VARjnp
    VARp = ws.VARp
    Bnp = ws.Bnp
    NBP = ws.NBP
    NNBP = ws.NNBP
    GP = ws.GP
    OM = ws.OM
    aux = ws.aux
    rhs = ws.rhs
    Xvec = ws.Xvec

    @inbounds for n in 1:N, j in 1:J
        VARjnp[j, n] = VARjn0[j, n] * om[j, n] * (Ljn_hat[j, n] ^ (1.0 - B[j, n]))
    end

    Chip = 0.0
    @inbounds for n in 1:N
        acc = 0.0
        for j in 1:J
            acc += VARjnp[j, n]
        end
        VARp[n] = acc
        Chip += acc
    end

    @inbounds for n in 1:N
        Bnp[n] = Snp[n] - io[n] * Chip + VARp[n]
    end

    @inbounds for importer in 1:N
        for dest in 1:N
            col0 = (dest - 1) * J
            for sec in 1:J
                rowidx = dest + (sec - 1) * N
                NBP[importer, col0 + sec] = Dinp[rowidx, importer]
            end
        end
    end

    @inbounds for importer in 1:N
        row0 = (importer - 1) * J
        for sec in 1:J
            row = row0 + sec
            for col in 1:JN
                NNBP[row, col] = NBP[importer, col]
            end
        end
    end

    @inbounds for idx in eachindex(GP)
        GP[idx] = ws.GG[idx] * NNBP[idx]
        OM[idx] = -GP[idx]
    end
    @inbounds for d in 1:JN
        OM[d, d] += 1.0
    end

    @inbounds for n in 1:N
        acc = 0.0
        for j in 1:J
            acc += om[j, n] * (Ljn_hat[j, n] ^ (1.0 - B[j, n])) * (VARjn0[j, n] + VALjn0[j, n])
        end
        aux[n] = acc - Bnp[n]
    end

    @inbounds for n in 1:N, j in 1:J
        rhs[j + (n - 1) * J] = alphas[j, n] * aux[n]
    end

    Xvec .= OM \ rhs
    reshape(Xvec, J, N)
end

function gmcnew!(omef0::Matrix{Float64}, ws::TempEqWorkspace, Xp::AbstractMatrix{Float64},
                 Dinp::Matrix{Float64}, J::Int, N::Int, R::Int, B::Matrix{Float64},
                 gamma::Matrix{Float64}, Ljn_hat::Matrix{Float64}, VARjn0::Matrix{Float64},
                 VALjn0::Matrix{Float64})
    JN = J * N
    PQ_vec = ws.PQ_vec
    DP = ws.DP
    Exjnp = ws.Exjnp
    aux4 = ws.aux4
    VAR = ws.VAR
    VAL = ws.VAL
    aux5 = ws.aux5

    @inbounds for j in 1:J, n in 1:N
        PQ_vec[n + (j - 1) * N] = Xp[j, n]
    end

    @inbounds for n in 1:N, idx in 1:JN
        DP[idx, n] = Dinp[idx, n] * PQ_vec[idx]
    end

    @inbounds for j in 1:J
        seg0 = (j - 1) * N
        for n in 1:N
            acc = 0.0
            for s in 1:N
                acc += DP[seg0 + s, n]
            end
            Exjnp[j, n] = acc
            aux4[j, n] = gamma[j, n] * acc
        end
    end

    @inbounds for n in 1:R, j in 1:J
        denom = (Ljn_hat[j, n] ^ (1.0 - B[j, n])) * (VARjn0[j, n] + VALjn0[j, n])
        omef0[j, n] = aux4[j, n] / denom
    end

    @inbounds for n in 1:N
        var_acc = 0.0
        val_acc = 0.0
        aux_acc = 0.0
        for j in 1:J
            var_acc += VARjn0[j, n]
            val_acc += VALjn0[j, n]
            aux_acc += aux4[j, n]
        end
        VAR[n] = var_acc
        VAL[n] = val_acc
        aux5[n] = aux_acc
    end

    @inbounds for n in (R + 1):N
        ratio = aux5[n] / (VAR[n] + VAL[n])
        for j in 1:J
            omef0[j, n] = ratio
        end
    end

    omef0
end

function solve_temporary_equilibrium_inplace!(base::BaseState4, Ljn_hat::Matrix{Float64},
                                              VARjn0::Matrix{Float64}, VALjn0::Matrix{Float64},
                                              Din::Matrix{Float64}, Snp::Vector{Float64};
                                              kappa_hat::Union{Nothing, Matrix{Float64}} = nothing,
                                              lambda_hat::Union{Nothing, Matrix{Float64}} = nothing,
                                              params::ModelParams = default_model_params(base),
                                              om_init::Union{Nothing, Matrix{Float64}} = nothing,
                                              workspace::Union{Nothing, TempEqWorkspace} = nothing,
                                              reset_price_guess::Bool = true)
    J, N, R = base.dims.J, base.dims.N, base.dims.R
    static_threaded = (params.profile == :reference) ? false : (params.use_threads && params.threads_static)
    ws = isnothing(workspace) ? TempEqWorkspace(base) : workspace

    kappa_hat_local = isnothing(kappa_hat) ? ones(J * N, N) : kappa_hat
    lambda_hat_local = isnothing(lambda_hat) ? ones(J, N) : lambda_hat
    kappa_is_one = _is_all_ones(kappa_hat_local)
    lambda_is_one = _is_all_ones(lambda_hat_local)
    om = ws.om
    if isnothing(om_init)
        fill!(om, 1.0)
    else
        om .= om_init
    end

    ommax = 1.0
    itw = 1

    while (itw <= params.max_iter_static) && (ommax > params.tol_static)
        phat, c = p_h_om!(
            ws,
            om,
            kappa_hat_local,
            lambda_hat_local,
            base.theta,
            base.G,
            base.gamma,
            Din;
            maxit = params.max_iter_static,
            tol = params.tol_static,
            threaded = static_threaded,
            kappa_is_one = kappa_is_one,
            lambda_is_one = lambda_is_one,
            reset_price_guess = reset_price_guess,
        )

        Dinp = dinprime!(
            ws,
            Din,
            kappa_hat_local,
            lambda_hat_local,
            c,
            phat,
            base.theta,
            base.gamma;
            threaded = static_threaded,
            kappa_is_one = kappa_is_one,
            lambda_is_one = lambda_is_one,
        )

        Xp = expenditurenew!(
            ws,
            J,
            N,
            base.alphas,
            base.B,
            Dinp,
            om,
            Ljn_hat,
            Snp,
            VARjn0,
            VALjn0,
            base.io,
        )

        gmcnew!(
            ws.om1,
            ws,
            Xp,
            Dinp,
            J,
            N,
            R,
            base.B,
            base.gamma,
            Ljn_hat,
            VARjn0,
            VALjn0,
        )

        @inbounds for idx in eachindex(om)
            zw = om[idx] - ws.om1[idx]
            ws.ZW[idx] = zw
            ws.om1[idx] = om[idx] * (1.0 + params.vfactor * zw / om[idx])
        end

        ommax = 0.0
        @inbounds for idx in 1:(J * R)
            d = ws.om1[idx] - om[idx]
            ommax += d * d
        end
        @inbounds for n in (R + 1):N
            idx = 1 + (n - 1) * J
            d = ws.om1[idx] - om[idx]
            ommax += d * d
        end

        om .= ws.om1
        itw += 1
    end

    wf0 = ws.wf
    rf0 = ws.rf
    @inbounds for n in 1:R, j in 1:J
        wf0[j, n] = om[j, n] * (Ljn_hat[j, n] ^ (-base.B[j, n]))
        rf0[j, n] = wf0[j, n] * Ljn_hat[j, n]
    end
    @inbounds for n in (R + 1):N, j in 1:J
        wf0[j, n] = om[j, n]
        rf0[j, n] = om[j, n]
    end

    VARjnp = ws.VARjnp
    VARp = ws.VARp
    Bnp = ws.Bnp
    @inbounds for n in 1:N, j in 1:J
        VARjnp[j, n] = VARjn0[j, n] * om[j, n] * (Ljn_hat[j, n] ^ (1.0 - base.B[j, n]))
    end

    Chip = 0.0
    @inbounds for n in 1:N
        acc = 0.0
        for j in 1:J
            acc += VARjnp[j, n]
        end
        VARp[n] = acc
        Chip += acc
    end
    @inbounds for n in 1:N
        Bnp[n] = Snp[n] - base.io[n] * Chip + VARp[n]
    end

    Dinp = ws.Dinp
    Xp = reshape(ws.Xvec, J, N)
    xbilat = ws.xbilat
    @inbounds for j in 1:J, n in 1:N
        ws.PQ_vec[n + (j - 1) * N] = Xp[j, n]
    end
    @inbounds for n in 1:N, idx in 1:(J * N)
        xbilat[idx, n] = ws.PQ_vec[idx] * Dinp[idx, n]
    end

    VALjnp = ws.VALjnp
    @inbounds for n in 1:N, j in 1:J
        VALjnp[j, n] = wf0[j, n] * Ljn_hat[j, n] * VALjn0[j, n]
    end

    phat_out = ws.pf0
    Pidx = ws.Pidx
    @inbounds for n in 1:N
        acc = 1.0
        for j in 1:J
            acc *= phat_out[j, n] ^ base.alphas[j, n]
        end
        Pidx[n] = acc
    end

    TempEqResultView(
        om,
        wf0,
        ws.VARjnp,
        VALjnp,
        Pidx,
        rf0,
        phat_out,
        ws.Dinp,
        reshape(ws.Xvec, J, N),
        ws.Bnp,
        ws.xbilat,
        ommax,
        itw - 1,
    )
end

function solve_temporary_equilibrium!(base::BaseState4, Ljn_hat::Matrix{Float64},
                                      VARjn0::Matrix{Float64}, VALjn0::Matrix{Float64},
                                      Din::Matrix{Float64}, Snp::Vector{Float64};
                                      kappa_hat::Union{Nothing, Matrix{Float64}} = nothing,
                                      lambda_hat::Union{Nothing, Matrix{Float64}} = nothing,
                                      params::ModelParams = default_model_params(base),
                                      om_init::Union{Nothing, Matrix{Float64}} = nothing,
                                      workspace::Union{Nothing, TempEqWorkspace} = nothing)
    temp = solve_temporary_equilibrium_inplace!(
        base,
        Ljn_hat,
        VARjn0,
        VALjn0,
        Din,
        Snp;
        kappa_hat = kappa_hat,
        lambda_hat = lambda_hat,
        params = params,
        om_init = om_init,
        workspace = workspace,
        reset_price_guess = true,
    )

    TempEqResult(
        copy(temp.om),
        copy(temp.wf),
        copy(temp.VARjn),
        copy(temp.VALjn),
        copy(temp.Pidx),
        copy(temp.rf),
        copy(temp.phat),
        copy(temp.Din),
        copy(temp.X),
        copy(temp.Sn),
        copy(temp.xbilat),
        temp.residual,
        temp.iterations,
    )
end
