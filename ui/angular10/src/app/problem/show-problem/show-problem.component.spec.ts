import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ShowProblemComponent } from './show-problem.component';

describe('ShowProblemComponent', () => {
  let component: ShowProblemComponent;
  let fixture: ComponentFixture<ShowProblemComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ShowProblemComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ShowProblemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
