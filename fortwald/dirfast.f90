SUBROUTINE dirvec( maxn, vec )
!
!  Calculates DIRect lattice VECtors (real space)
! for ewald summation, 
!   k = -k, hence -k is neglected
!
  INTEGER :: maxn     ! maximal abs(n_i)
  INTEGER :: x, y, z  ! n = (n_x, n_y, n_z)
  INTEGER :: cnt      ! counter
  INTEGER :: vec(maxn * (2 * maxn + 1) * (2 * maxn + 1) + & 
    & (2 * maxn + 1) * (2 * maxn + 2) / 2 - maxn, 3)
!f2py intent(in) maxn
!f2py intent(out) vec
!f2py depend(maxn) vec
  cnt = 1
  z = 0
  DO y = 1, maxn
    DO x = -y+1, maxn
      vec(cnt, 1) = x
      vec(cnt, 2) = y
      vec(cnt, 3) = z
      cnt = cnt + 1
    ENDDO
  ENDDO
  DO y = -maxn, 0
    DO x = -y, maxn
      vec(cnt, 1) = x
      vec(cnt, 2) = y
      vec(cnt, 3) = z
      cnt = cnt + 1
    ENDDO
  ENDDO
  DO z = 1, maxn
    DO y = -maxn, maxn
      DO x = -maxn, maxn
        vec(cnt, 1) = x
        vec(cnt, 2) = y
        vec(cnt, 3) = z
        cnt = cnt + 1
      ENDDO
    ENDDO
  ENDDO
END SUBROUTINE dirvec

