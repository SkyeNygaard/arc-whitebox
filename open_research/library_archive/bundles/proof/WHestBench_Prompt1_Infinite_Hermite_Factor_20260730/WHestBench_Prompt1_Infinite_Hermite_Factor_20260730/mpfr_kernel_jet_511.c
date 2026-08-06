#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Minimal ABI declarations for MPFR 4.x (dev headers unavailable). */
typedef long mpfr_prec_t;
typedef int mpfr_sign_t;
typedef long mpfr_exp_t;
typedef unsigned long mp_limb_t;
typedef int mpfr_rnd_t;
typedef struct { mpfr_prec_t _mpfr_prec; mpfr_sign_t _mpfr_sign; mpfr_exp_t _mpfr_exp; mp_limb_t *_mpfr_d; } __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef const __mpfr_struct *mpfr_srcptr;
typedef __mpfr_struct *mpfr_ptr;
extern void mpfr_init2(mpfr_ptr, mpfr_prec_t);
extern void mpfr_clear(mpfr_ptr);
extern int mpfr_set(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_set_ui(mpfr_ptr, unsigned long, mpfr_rnd_t);
extern int mpfr_set_si(mpfr_ptr, long, mpfr_rnd_t);
extern int mpfr_set_str(mpfr_ptr, const char*, int, mpfr_rnd_t);
extern int mpfr_add(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_sub(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_mul(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_div(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_mul_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
extern int mpfr_div_ui(mpfr_ptr, mpfr_srcptr, unsigned long, mpfr_rnd_t);
extern int mpfr_neg(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_sqrt(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_acos(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
extern int mpfr_const_pi(mpfr_ptr, mpfr_rnd_t);
extern int mpfr_cmp(mpfr_srcptr, mpfr_srcptr);
extern int mpfr_zero_p(mpfr_srcptr);
extern size_t __gmpfr_out_str(FILE*, int, size_t, mpfr_srcptr, mpfr_rnd_t);

#define RNDN 0
#define RNDU 2
#define RNDD 3
#define ORDER 511
#define DEPTH 32
#define PREC 320

typedef struct { mpfr_t lo, hi; } I;
typedef struct { I c[ORDER+1]; } S;

static void ii(I *x){ mpfr_init2(x->lo,PREC); mpfr_init2(x->hi,PREC); mpfr_set_ui(x->lo,0,RNDN); mpfr_set_ui(x->hi,0,RNDN); }
static void ic(I *x){ mpfr_clear(x->lo); mpfr_clear(x->hi); }
static void iz(I *x){mpfr_set_ui(x->lo,0,RNDN);mpfr_set_ui(x->hi,0,RNDN);}
static void iui(I*x,unsigned long a){mpfr_set_ui(x->lo,a,RNDN);mpfr_set_ui(x->hi,a,RNDN);}
static void isi(I*x,long a){mpfr_set_si(x->lo,a,RNDN);mpfr_set_si(x->hi,a,RNDN);}
static void icopy(I*o,const I*a){mpfr_set(o->lo,a->lo,RNDD);mpfr_set(o->hi,a->hi,RNDU);}
static void iadd(I*o,const I*a,const I*b){mpfr_add(o->lo,a->lo,b->lo,RNDD);mpfr_add(o->hi,a->hi,b->hi,RNDU);}
static void isub(I*o,const I*a,const I*b){mpfr_sub(o->lo,a->lo,b->hi,RNDD);mpfr_sub(o->hi,a->hi,b->lo,RNDU);}
static void ineg(I*o,const I*a){mpfr_neg(o->lo,a->hi,RNDD);mpfr_neg(o->hi,a->lo,RNDU);}
static void minset(mpfr_ptr o, mpfr_srcptr a, mpfr_srcptr b){mpfr_set(o, mpfr_cmp(a,b)<=0?a:b, RNDN);}
static void maxset(mpfr_ptr o, mpfr_srcptr a, mpfr_srcptr b){mpfr_set(o, mpfr_cmp(a,b)>=0?a:b, RNDN);}
static void imul(I*o,const I*a,const I*b){
 mpfr_t p1,p2,p3,p4,m1,m2; mpfr_init2(p1,PREC);mpfr_init2(p2,PREC);mpfr_init2(p3,PREC);mpfr_init2(p4,PREC);mpfr_init2(m1,PREC);mpfr_init2(m2,PREC);
 mpfr_mul(p1,a->lo,b->lo,RNDD); mpfr_mul(p2,a->lo,b->hi,RNDD); mpfr_mul(p3,a->hi,b->lo,RNDD); mpfr_mul(p4,a->hi,b->hi,RNDD);
 minset(m1,p1,p2);minset(m2,p3,p4);minset(o->lo,m1,m2);
 mpfr_mul(p1,a->lo,b->lo,RNDU); mpfr_mul(p2,a->lo,b->hi,RNDU); mpfr_mul(p3,a->hi,b->lo,RNDU); mpfr_mul(p4,a->hi,b->hi,RNDU);
 maxset(m1,p1,p2);maxset(m2,p3,p4);maxset(o->hi,m1,m2);
 mpfr_clear(p1);mpfr_clear(p2);mpfr_clear(p3);mpfr_clear(p4);mpfr_clear(m1);mpfr_clear(m2);
}
static int contains0(const I*a){mpfr_t z;mpfr_init2(z,32);mpfr_set_ui(z,0,RNDN);int r=mpfr_cmp(a->lo,z)<=0&&mpfr_cmp(a->hi,z)>=0;mpfr_clear(z);return r;}
static void iinv(I*o,const I*a){ if(contains0(a)){fprintf(stderr,"division by interval containing zero\n");exit(2);} mpfr_t one;mpfr_init2(one,PREC);mpfr_set_ui(one,1,RNDN); if(mpfr_cmp(a->lo,one)>0 || mpfr_cmp(a->hi,one)<0 || 1){mpfr_div(o->lo,one,a->hi,RNDD);mpfr_div(o->hi,one,a->lo,RNDU);} mpfr_clear(one); }
static void idiv(I*o,const I*a,const I*b){I inv;ii(&inv);iinv(&inv,b);imul(o,a,&inv);ic(&inv);}
static void imului(I*o,const I*a,unsigned long n){mpfr_mul_ui(o->lo,a->lo,n,RNDD);mpfr_mul_ui(o->hi,a->hi,n,RNDU);}
static void idivui(I*o,const I*a,unsigned long n){mpfr_div_ui(o->lo,a->lo,n,RNDD);mpfr_div_ui(o->hi,a->hi,n,RNDU);}
static void isqrt(I*o,const I*a){mpfr_sqrt(o->lo,a->lo,RNDD);mpfr_sqrt(o->hi,a->hi,RNDU);}
static void iacos(I*o,const I*a){mpfr_acos(o->lo,a->hi,RNDD);mpfr_acos(o->hi,a->lo,RNDU);}
static void ipi(I*o){mpfr_const_pi(o->lo,RNDD);mpfr_const_pi(o->hi,RNDU);}
static void sinit(S*s){for(int i=0;i<=ORDER;i++)ii(&s->c[i]);}
static void sclear(S*s){for(int i=0;i<=ORDER;i++)ic(&s->c[i]);}
static void szero(S*s){for(int i=0;i<=ORDER;i++)iz(&s->c[i]);}
static void scopy(S*o,const S*a){for(int i=0;i<=ORDER;i++)icopy(&o->c[i],&a->c[i]);}
static void sadd(S*o,const S*a,const S*b){for(int i=0;i<=ORDER;i++)iadd(&o->c[i],&a->c[i],&b->c[i]);}
static void sneg(S*o,const S*a){for(int i=0;i<=ORDER;i++)ineg(&o->c[i],&a->c[i]);}
static void sscale(S*o,const S*a,const I*k){for(int i=0;i<=ORDER;i++)imul(&o->c[i],&a->c[i],k);}
static void smul(S*o,const S*a,const S*b){
 S t;sinit(&t);szero(&t);I p,sum;ii(&p);ii(&sum);
 for(int n=0;n<=ORDER;n++){iz(&sum);for(int i=0;i<=n;i++){imul(&p,&a->c[i],&b->c[n-i]);iadd(&sum,&sum,&p);}icopy(&t.c[n],&sum);}scopy(o,&t);ic(&p);ic(&sum);sclear(&t);
}
static void sinv(S*o,const S*a){
 S t;sinit(&t);szero(&t);I one,sum,p,neg;ii(&one);iui(&one,1);ii(&sum);ii(&p);ii(&neg);idiv(&t.c[0],&one,&a->c[0]);
 for(int n=1;n<=ORDER;n++){iz(&sum);for(int i=1;i<=n;i++){imul(&p,&a->c[i],&t.c[n-i]);iadd(&sum,&sum,&p);}ineg(&neg,&sum);idiv(&t.c[n],&neg,&a->c[0]);}
 scopy(o,&t);ic(&one);ic(&sum);ic(&p);ic(&neg);sclear(&t);
}
static void ssqrt(S*o,const S*a){
 S t;sinit(&t);szero(&t);I sum,p,num,den;ii(&sum);ii(&p);ii(&num);ii(&den);isqrt(&t.c[0],&a->c[0]);
 for(int n=1;n<=ORDER;n++){iz(&sum);for(int i=1;i<n;i++){imul(&p,&t.c[i],&t.c[n-i]);iadd(&sum,&sum,&p);}isub(&num,&a->c[n],&sum);imului(&den,&t.c[0],2);idiv(&t.c[n],&num,&den);}
 scopy(o,&t);ic(&sum);ic(&p);ic(&num);ic(&den);sclear(&t);
}
static void sder(S*o,const S*a){szero(o);for(int n=0;n<ORDER;n++)imului(&o->c[n],&a->c[n+1],n+1);}
static void sint(S*o,const S*a,const I*constant){szero(o);icopy(&o->c[0],constant);for(int n=1;n<=ORDER;n++)idivui(&o->c[n],&a->c[n-1],n);}
static void relu(S*out,const S*p){
 S one,p2,om,root,dp,rinv,prod,ader,aser,negas,pias,mul,sum; S *arr[]={&one,&p2,&om,&root,&dp,&rinv,&prod,&ader,&aser,&negas,&pias,&mul,&sum};for(size_t i=0;i<sizeof(arr)/sizeof(arr[0]);i++){sinit(arr[i]);szero(arr[i]);}
 iui(&one.c[0],1);smul(&p2,p,p);for(int i=0;i<=ORDER;i++)isub(&om.c[i],&one.c[i],&p2.c[i]);ssqrt(&root,&om);sder(&dp,p);sinv(&rinv,&root);smul(&prod,&dp,&rinv);for(int i=0;i<=ORDER;i++)ineg(&ader.c[i],&prod.c[i]);
 I ac;ii(&ac);iacos(&ac,&p->c[0]);sint(&aser,&ader,&ac);sneg(&negas,&aser);I pi;ii(&pi);ipi(&pi);iadd(&pias.c[0],&negas.c[0],&pi);for(int i=1;i<=ORDER;i++)icopy(&pias.c[i],&negas.c[i]);smul(&mul,&pias,p);sadd(&sum,&root,&mul);I invpi;ii(&invpi);iinv(&invpi,&pi);sscale(out,&sum,&invpi);
 ic(&ac);ic(&pi);ic(&invpi);for(size_t i=0;i<sizeof(arr)/sizeof(arr[0]);i++)sclear(arr[i]);
}
int main(){
 S p,nxt;sinit(&p);sinit(&nxt);szero(&p);iui(&p.c[1],1);
 for(int d=0;d<DEPTH;d++){relu(&nxt,&p);scopy(&p,&nxt);}
 printf("{\n  \"precision_bits\": %d,\n  \"order\": %d,\n  \"depth\": %d,\n  \"coefficients\": [\n",PREC,ORDER,DEPTH);
 for(int i=0;i<=ORDER;i++){
  printf("    {\"degree\":%d,\"lo\":\"",i);__gmpfr_out_str(stdout,10,0,p.c[i].lo,RNDD);printf("\",\"hi\":\"");__gmpfr_out_str(stdout,10,0,p.c[i].hi,RNDU);printf("\"}%s\n",i==ORDER?"":",");
 }
 printf("  ]\n}\n");sclear(&p);sclear(&nxt);return 0;
}
