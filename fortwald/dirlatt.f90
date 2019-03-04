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
      CALL Vec3D(vec(cnt, :), cnt, x, y, z)
    ENDDO
  ENDDO
  DO y = -maxn, 0
    DO x = -y, maxn
      CALL Vec3D(vec(cnt, :), cnt, x, y, z)
    ENDDO
  ENDDO
  DO z = 1, maxn
    DO y = -maxn, maxn
      DO x = -maxn, maxn
        CALL Vec3D(vec(cnt, :), cnt, x, y, z)
      ENDDO
    ENDDO
  ENDDO
END SUBROUTINE dirvec

SUBROUTINE Vec3D( v, i, vx, vy, vz )
  INTEGER, INTENT(INOUT) :: v(3)
  INTEGER, INTENT(INOUT) :: i
  INTEGER, INTENT(IN) :: vx, vy, vz
  v(1) = vx
  v(2) = vy
  v(3) = vz
  i = i + 1
END SUBROUTINE Vec3D

